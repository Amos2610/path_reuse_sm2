#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.time import Time as RclpyTime
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.duration import Duration as RclpyDuration
import tf2_ros
import tf2_geometry_msgs
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from typing import Any, List, Optional
from xarm_utils_py import XArmUtils, Node


class XArmUtilsWrapper:
    """任意スキルから継承して xarm をそのまま使える薄い基底クラス"""
    
    xarm_node = None
    _xarm = None
    _node_name = "xarm_core_node"
    _arm_name = "xarm6"

    def __init__(self):
        if XArmUtilsWrapper.xarm_node is None:
            XArmUtilsWrapper.xarm_node = Node(XArmUtilsWrapper._node_name)
        if XArmUtilsWrapper._xarm is None:
            XArmUtilsWrapper._xarm = XArmUtils(XArmUtilsWrapper.xarm_node, XArmUtilsWrapper._arm_name)

        # インスタンスから参照できるように設定
        self.xarm_node = XArmUtilsWrapper.xarm_node
        self.xarm = XArmUtilsWrapper._xarm

    def set_pipeline(self, name: str):
        self.xarm.set_planning_pipeline(name)

    def set_move_group_parameter(self, key: str, value: Any):
        self.xarm.set_move_group_parameter(key, value)


class XArmRobotUtils:
    """xArm向けのPose，Joint変換ユーティリティ．"""

    def __init__(
        self,
        node,
        base_frame="link_base",
        ee_link="link_eef",
        move_group="xarm6",
        ik_service="/compute_ik",
        xarm=None,
        default_pose_frame=None,
    ):
        self.node = node
        self.base_frame = base_frame
        self.ee_link = ee_link
        self.move_group = move_group
        self.ik_service = ik_service
        self.xarm = xarm
        self.default_pose_frame = default_pose_frame or base_frame
        self._ik_cli = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

    def _is_sequence(self, value):
        if isinstance(value, (str, bytes, dict)):
            return False
        return hasattr(value, "__len__") and hasattr(value, "__getitem__")

    def is_joint_list(self, value):
        if not self._is_sequence(value):
            return False
        if len(value) == 7:
            return False
        if len(value) < 6:
            return False
        try:
            [float(v) for v in list(value)[:6]]
            return True
        except Exception:
            return False

    def normalize_joint_list(self, value):
        if not self.is_joint_list(value):
            return None
        return [float(v) for v in list(value)[:6]]

    def pose_to_pose_stamped(self, pose):
        # TODO(real-robot): transform non-base-frame pose to base_frame using TF before IK.
        # Current implementation assumes pose is already expressed in base_frame.
        if pose is None:
            return None

        if hasattr(pose, "header") and hasattr(pose, "pose"):
            return pose

        ps = PoseStamped()
        ps.header.frame_id = self.default_pose_frame
        ps.header.stamp = RclpyTime().to_msg()  # time=0: use latest available TF

        self.node.get_logger().info(f"[XArmRobotUtils] pose_to_pose_stamped input: {pose}")
        if self._is_sequence(pose) and len(pose) == 7:
            pose = list(pose)
            ps.pose.position.x = float(pose[0])
            ps.pose.position.y = float(pose[1])
            ps.pose.position.z = float(pose[2])
            ps.pose.orientation.x = float(pose[3])
            ps.pose.orientation.y = float(pose[4])
            ps.pose.orientation.z = float(pose[5])
            ps.pose.orientation.w = float(pose[6])
            return ps

        if isinstance(pose, dict):
            if "pose" in pose and isinstance(pose["pose"], dict):
                pose = pose["pose"]

            position = pose.get("position") or {}
            orientation = pose.get("orientation") or {}

            ps.header.frame_id = pose.get("frame_id") or self.default_pose_frame
            ps.pose.position.x = float(position.get("x", 0.0))
            ps.pose.position.y = float(position.get("y", 0.0))
            ps.pose.position.z = float(position.get("z", 0.0))
            ps.pose.orientation.x = float(orientation.get("x", 0.0))
            ps.pose.orientation.y = float(orientation.get("y", 0.0))
            ps.pose.orientation.z = float(orientation.get("z", 0.0))
            ps.pose.orientation.w = float(orientation.get("w", 1.0))
            return ps

        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            ps.pose = pose
            return ps

        return None

    def _transform_pose_to_base(self, pose_stamped):
        if pose_stamped is None:
            return None

        src_frame = pose_stamped.header.frame_id or self.default_pose_frame
        pose_stamped.header.frame_id = src_frame

        if src_frame == self.base_frame:
            return pose_stamped

        p = pose_stamped.pose.position
        q = pose_stamped.pose.orientation
        self.node.get_logger().info(
            f"[XArmRobotUtils] raw pose [{src_frame}]: "
            f"pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  "
            f"quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})"
        )

        try:
            transformed = self._tf_buffer.transform(
                pose_stamped,
                self.base_frame,
                timeout=RclpyDuration(seconds=0.5),
            )
            tp = transformed.pose.position
            self.node.get_logger().info(
                f"[XArmRobotUtils] transformed pose [{src_frame} -> {self.base_frame}]: "
                f"pos=({tp.x:.4f}, {tp.y:.4f}, {tp.z:.4f})"
            )
            return transformed
        except Exception as e:
            self.node.get_logger().error(
                f"[XArmRobotUtils] TF transform failed: {src_frame} -> {self.base_frame}: {e}"
            )
            return None

    def compute_ik(self, pose, seed_joints: Optional[List[float]] = None):
        target = self.pose_to_pose_stamped(pose)
        target = self._transform_pose_to_base(target)
        if target is None:
            return None
        
        # baseのz座標は決め打ち
        target.pose.position.z = 0.125

        if self._ik_cli is None:
            self._ik_cli = self.node.create_client(GetPositionIK, self.ik_service)

        if not self._ik_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error(f"[XArmRobotUtils] IK service not available: {self.ik_service}")
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.move_group
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped = target
        req.ik_request.timeout = DurationMsg(sec=1, nanosec=0)
        req.ik_request.avoid_collisions = False

        # seed_joints をセットすると姿勢の初期ジョイントがセットできる
        if seed_joints is not None and len(seed_joints) >= 6:
            js = JointState()
            js.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            js.position = [float(v) for v in seed_joints[:6]]
            rs = RobotState()
            rs.joint_state = js
            req.ik_request.robot_state = rs

        future = self._ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        res = future.result()

        if res is None:
            self.node.get_logger().error("[XArmRobotUtils] IK service call failed")
            return None

        if res.error_code.val != res.error_code.SUCCESS:
            self.node.get_logger().error(f"[XArmRobotUtils] IK failed, error_code={res.error_code.val}")
            return None

        joints = list(res.solution.joint_state.position)[:6]
        self.node.get_logger().info(f"[XArmRobotUtils] IK joints: {joints}")
        return joints

    def get_current_joint_values(self):
        # TODO(real-robot): replace fallback behavior with actual current joint acquisition.
        # In fake MoveIt, current_state can be empty or timestamp 0, so returning None is expected.
        candidates = []

        if self.xarm is not None:
            candidates.append(self.xarm)
            for attr in [
                "move_group",
                "group",
                "_move_group",
                "move_group_interface",
                "robot",
            ]:
                obj = getattr(self.xarm, attr, None)
                if obj is not None:
                    candidates.append(obj)

        for obj in candidates:
            for method_name in [
                "get_current_joint_values",
                "get_joint_values",
                "get_current_joints",
                "get_joints",
            ]:
                fn = getattr(obj, method_name, None)
                if fn is None:
                    continue
                try:
                    joints = fn()
                except Exception as e:
                    self.node.get_logger().warn(
                        f"[XArmRobotUtils] failed to call {method_name}: {e}"
                    )
                    continue

                joints = self.normalize_joint_list(joints)
                if joints:
                    self.node.get_logger().info(
                        f"[XArmRobotUtils] current joint values: {joints}"
                    )
                    return joints

        self.node.get_logger().warn(
            "[XArmRobotUtils] current joint values are unavailable or empty."
        )
        return None

    def transform_to_base(self, pose):
        """Transform a pose to the base frame, returning a PoseStamped. Returns None on failure."""
        ps = self.pose_to_pose_stamped(pose)
        if ps is None:
            return None
        return self._transform_pose_to_base(ps)

    def resolve_joints(self, *candidates):
        for value in candidates:
            joints = self.normalize_joint_list(value)
            if joints:
                return joints

            pose = self.pose_to_pose_stamped(value)
            if pose is not None:
                joints = self.compute_ik(pose)
                joints = self.normalize_joint_list(joints)
                if joints:
                    return joints

        return None

