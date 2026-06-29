#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Optional
import time
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from yasmin.state import State
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import qos_profile_sensor_data

try:
    from cv_bridge import CvBridge, CvBridgeError
    _cv_bridge_available = True
except ImportError:
    _cv_bridge_available = False

try:
    from image_geometry import PinholeCameraModel
    _image_geometry_available = True
except ImportError:
    _image_geometry_available = False


class FindObj(State):
    """
    物体検出ステート。
    RAG/KB から認識結果（obj_joints / grasp_pose）が提供済みであれば検出をスキップする。
    検出は将来 GroundedSAM2 等で実装予定。
    """

    def __init__(self, node, **kwargs):
        super().__init__(outcomes=["success", "except", "loop"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.workpiece = kwargs.get("workpiece") or kwargs.get("target") or ""
        self.step_index = kwargs.get("step_index", 0)

        # --- カメラ関連（将来の検出実装用） ---
        qos = qos_profile_sensor_data
        self.image_topic = kwargs.get("image_topic", "/camera/hand_camera/color/image_raw")
        self.camera_info_topic = kwargs.get("camera_info_topic", "/camera/hand_camera/color/camera_info")
        self.camera_info_wait_timeout = float(kwargs.get("camera_info_wait_timeout", 2.0))
        self.camera_info_wait_poll_sec = float(kwargs.get("camera_info_wait_poll_sec", 0.05))

        self.image_msg: Optional[Image] = None
        self.cv2_image = None
        self._logged_camera_info = False

        self.bridge = CvBridge() if _cv_bridge_available else None
        self.camera_model = PinholeCameraModel() if _image_geometry_available else None
        self.camera_frame_id: str = ""
        self.info_msg: Optional[CameraInfo] = None

        self.image_sub = self.pr_node.create_subscription(
            Image, self.image_topic, self._image_callback, qos
        )
        self.info_sub = self.pr_node.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback, qos
        )
        self.pr_node.get_logger().info(
            f"[FindObj][frame] subscribing camera_info_topic={self.camera_info_topic}, "
            f"initial camera_frame_id={repr(self.camera_frame_id)}"
        )

        # --- TF（将来の pixel→3D 変換用） ---
        self.base_frame = kwargs.get("base_frame", "link_base")
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.pr_node)

    # ------------------------------------------------------------------
    # コールバック
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        self.image_msg = msg
        if self.bridge is None:
            return
        try:
            self.cv2_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.cv2_image = None
            self.pr_node.get_logger().warn(f"[FindObj] cv_bridge error: {e}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.info_msg = msg
        self.camera_frame_id = msg.header.frame_id
        if self.camera_model is not None:
            try:
                self.camera_model.fromCameraInfo(msg)
            except Exception as e:
                self.pr_node.get_logger().warn(f"[FindObj] PinholeCameraModel update failed: {e}")
        if not self._logged_camera_info:
            self.pr_node.get_logger().info(
                f"[FindObj] camera_info: frame={msg.header.frame_id}, "
                f"size=({msg.width}x{msg.height})"
            )
            self._logged_camera_info = True

    def _wait_for_camera_info(self) -> None:
        if self.info_msg is not None and self.camera_frame_id:
            return

        if self.camera_info_wait_timeout <= 0.0:
            return

        self.pr_node.get_logger().info(
            f"[FindObj][frame] waiting for CameraInfo on {self.camera_info_topic} "
            f"up to {self.camera_info_wait_timeout:.2f}s"
        )
        deadline = time.monotonic() + self.camera_info_wait_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if self.info_msg is not None and self.camera_frame_id:
                self.pr_node.get_logger().info(
                    f"[FindObj][frame] CameraInfo received while waiting: "
                    f"frame_id={repr(self.camera_frame_id)}"
                )
                return
            time.sleep(self.camera_info_wait_poll_sec)

        self.pr_node.get_logger().warn(
            f"[FindObj][frame] timed out waiting for CameraInfo on {self.camera_info_topic}. "
            f"info_received={self.info_msg is not None}, camera_frame_id={repr(self.camera_frame_id)}"
        )

    # ------------------------------------------------------------------
    # メイン処理
    # ------------------------------------------------------------------

    def execute(self, blackboard: Any = None) -> str:
        self.pr_node.get_logger().info("[FindObj] execute")

        def _bb_get(key, default=None):
            try:
                return blackboard[key]
            except Exception:
                return default

        def _bb_set(key, value):
            try:
                blackboard[key] = value
            except Exception:
                setattr(blackboard, key, value)

        resolved_workpiece = _bb_get("workpiece", None) or self.workpiece
        if resolved_workpiece:
            self.workpiece = resolved_workpiece
            _bb_set("workpiece", resolved_workpiece)

        self._wait_for_camera_info()

        bb_camera_frame_id = _bb_get("camera_frame_id", "")
        bb_grasp_pose_frame_id = _bb_get("grasp_pose_frame_id", "")
        self.pr_node.get_logger().info(
            f"[FindObj][frame] execute current camera_frame_id={repr(self.camera_frame_id)}, "
            f"info_received={self.info_msg is not None}, "
            f"bb.camera_frame_id={repr(bb_camera_frame_id)}, "
            f"bb.grasp_pose_frame_id={repr(bb_grasp_pose_frame_id)}"
        )

        if self.camera_frame_id:
            _bb_set("camera_frame_id", self.camera_frame_id)
            _bb_set("grasp_pose_frame_id", self.camera_frame_id)
            self.pr_node.get_logger().info(
                f"[FindObj][frame] exported to blackboard: camera_frame_id={repr(self.camera_frame_id)}"
            )
        else:
            self.pr_node.get_logger().warn(
                f"[FindObj][frame] camera_frame_id is empty. "
                f"No CameraInfo has been received from {self.camera_info_topic} before execute."
            )

        # RAG/KB から認識結果が既に得られていれば検出をスキップ
        bb_grasp_pose = _bb_get("grasp_pose", None)
        bb_obj_joints = _bb_get("obj_joints", None)
        has_joints = isinstance(bb_obj_joints, (list, tuple)) and len(bb_obj_joints) >= 6
        has_pose = bb_grasp_pose is not None
        self.pr_node.get_logger().info(
            f"[FindObj] workpiece={repr(self.workpiece)}, "
            f"has_joints={has_joints}, has_pose={has_pose}"
        )
        if has_joints or has_pose:
            self.pr_node.get_logger().info(
                "[FindObj] Recognition result from RAG/KB. Skipping detection."
            )
            if hasattr(self, 'image_sub') and self.image_sub is not None:
                self.pr_node.destroy_subscription(self.image_sub)
                self.image_sub = None
            if hasattr(self, 'info_sub') and self.info_sub is not None:
                self.pr_node.destroy_subscription(self.info_sub)
                self.info_sub = None
            if hasattr(self, 'tf_listener') and self.tf_listener is not None:
                self.tf_listener = None
            return "success"

        # --- cleanup subscriptions to avoid leak ---
        if hasattr(self, 'image_sub') and self.image_sub is not None:
            self.pr_node.destroy_subscription(self.image_sub)
            self.image_sub = None
        if hasattr(self, 'info_sub') and self.info_sub is not None:
            self.pr_node.destroy_subscription(self.info_sub)
            self.info_sub = None
        if hasattr(self, 'tf_listener') and self.tf_listener is not None:
            self.tf_listener = None

        # TODO: GroundedSAM2 による検出を実装する
        #
        # 検出後は以下のように grasp_pose を blackboard に書き込む:
        # grasp_pose_dict = {
        #     "frame_id": self.camera_frame_id,   # 例: "camera_color_optical_frame"
        #     "position": {"x": x, "y": y, "z": z},
        #     "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        # }
        # _bb_set("grasp_pose", grasp_pose_dict)

        return "success"
