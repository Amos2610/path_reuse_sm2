#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import List, Any, Optional, Tuple
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, PoseStamped
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import Pose, Point, Quaternion
from yasmin.state import State
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import qos_profile_sensor_data   
try:
    from image_geometry import PinholeCameraModel
except ImportError as e:
    PinholeCameraModel = None  # image_geometryが利用できない場合のフォールバック
from ultralytics import YOLO


class FindObj(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=["success", "except", "loop"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.pr_client = PathSeedClient()
        self.target_location = kwargs.get("target_location", "")
        # LangGraphまたはTaskSetから渡されたworkpieceを使う．
        # ここに値がある場合，execute側で検出済みとして扱える．
        self.workpiece = kwargs.get("workpiece") or kwargs.get("target") or ""
        self.path_seed_path = kwargs.get("path_seed_path", "")
        self.step_index = kwargs.get("step_index", 0)

        # YOLOの初期設定
        self.model = None
        self.yolo_model_path = kwargs.get("yolo_model", "yolo26x-seg.pt")
        try:
            self.model = YOLO(self.yolo_model_path)
            self.pr_node.get_logger().info(f"[FindObj] YOLO model loaded: {self.yolo_model_path}")
        except Exception as e:
            self.pr_node.get_logger().warn(f"[FindObj] YOLO model could not be loaded: {e}")
        self.conf_th = float(kwargs.get("conf_th", 0.25)) # 信頼度
        self.iou_th = float(kwargs.get("iou_th", 0.45))

        # Subscribers
        qos = qos_profile_sensor_data
        self.bridge = CvBridge()
        self.image_msg: Optional[Image] = None
        self.depth_msg: Optional[Image] = None 
        self.cv2_image = None  # np.ndarray (BGR)
        self.cv2_depth = None  # np.ndarray (Depth)
        self._p_max_distance = -1.0  # 最大距離制限なし
        self.approach_offset_z = float(kwargs.get("approach_offset_z", 0.05)) if isinstance(kwargs, dict) else 0.05

        self.image_sub = self.pr_node.create_subscription(
            Image,
            "/camera/camera/color/image_raw", # realsenseD435
            self.image_callback,
            qos,
        )
        self.depth_sub = self.pr_node.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw", # realsenseD435
            self.depth_callback,
            qos,
        )
        self.info_msg: Optional[CameraInfo] = None
        self.camera_frame_id: str = ""

        self.info_sub = self.pr_node.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",  # realsenseD435
            self.camera_info_callback,
            qos,
        )

        self.yolo_pub = self.pr_node.create_publisher(
            Image,
            "/object_image",
            10,
        )
    
        try:
            self.camera_model = PinholeCameraModel()
        except Exception as e:
            self.camera_model = None
            self.pr_node.get_logger().warn(f"[FindObj] PinholeCameraModel could not be initialized: {e}")
        # TF (camera -> base) + IK service client
        self.base_frame = kwargs.get("base_frame", "link_base")
        self.ee_link = kwargs.get("ee_link", "link_eef")
        self.move_group_name = kwargs.get("move_group", "xarm6")

        # TF listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.pr_node)

        # MoveIt IK service
        self.ik_service_name = kwargs.get("ik_service", "/compute_ik")
        self.ik_cli = self.pr_node.create_client(GetPositionIK, self.ik_service_name)

    def image_callback(self, msg: Image):
        """msg: sensor_msgs/Image を受け取って cv2 画像(BGR)に変換して保持"""
        self.image_msg = msg
        try:
            # ROS Image -> OpenCV (BGR)
            self.cv2_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.cv2_image = None
            self.pr_node.get_logger().warn(f"[FindObj] CvBridgeError: {e}")
    
    def depth_callback(self, msg: Image):
        """msg: sensor_msgs/Image を受け取って cv2 画像(Depth)に変換して保持"""
        self.depth_msg = msg
        try:
            # ROS Image -> OpenCV (Depth)
            self.cv2_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.cv2_depth = None
            self.pr_node.get_logger().warn(f"[FindObj] CvBridgeError: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        """CameraInfo を受け取って PinholeCameraModel を初期化"""
        self.info_msg = msg
        try:
            self.set_camera_model(msg)
        except Exception as e:
            self.pr_node.get_logger().warn(f"[FindObj] set_camera_model failed: {e}")

    def _camera_point_to_base(self, x_cam: float, y_cam: float, z_cam: float) -> Optional[Tuple[float, float, float]]:
        """
        camera座標系の点 (x,y,z) [m] を TF で base座標系へ変換して返す
        """
        # camera_frame_id は CameraInfo から設定される
        src_frame = getattr(self, "camera_frame_id", "") or ""
        if not src_frame:
            self.pr_node.get_logger().error("[FindObj] camera_frame_id is empty (CameraInfo not received?)")
            return None

        try:
            # 最新のTFを取得
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,   # target
                src_frame,         # source
                Time(),            # latest
                timeout=Duration(seconds=0.2),
            )

            p = PointStamped()
            p.header.frame_id = src_frame
            p.header.stamp = self.pr_node.get_clock().now().to_msg()
            p.point.x = float(x_cam)
            p.point.y = float(y_cam)
            p.point.z = float(z_cam)

            p_base = do_transform_point(p, tf)
            return (p_base.point.x, p_base.point.y, p_base.point.z)

        except Exception as e:
            self.pr_node.get_logger().error(f"[FindObj] TF lookup/transform failed: {e}")
            return None

    def _get_current_ee_pose_in_base(self) -> Optional[PoseStamped]:
        """TFで base_frame -> ee_link の現在姿勢を取得"""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,   # target
                self.ee_link,      # source
                Time(),            # latest
                timeout=Duration(seconds=0.2),
            )

            ps = PoseStamped()
            ps.header.frame_id = self.base_frame
            ps.header.stamp = self.pr_node.get_clock().now().to_msg()
            ps.pose.position.x = tf.transform.translation.x
            ps.pose.position.y = tf.transform.translation.y
            ps.pose.position.z = tf.transform.translation.z
            ps.pose.orientation = tf.transform.rotation
            return ps
        except Exception as e:
            self.pr_node.get_logger().error(f"[FindObj] TF get current ee pose failed: {e}")
            return None

    def _compute_ik(self, target_pose: PoseStamped) -> Optional[list]:
        """
        MoveItのIKサービス /compute_ik を呼んで関節角(list[float])を返す
        """
        if not self.ik_cli.wait_for_service(timeout_sec=0.5):
            self.pr_node.get_logger().error(f"[FindObj] IK service not available: {self.ik_service_name}")
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.move_group_name
        req.ik_request.pose_stamped = target_pose
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.timeout = Duration(seconds=0.5).to_msg()
        req.ik_request.avoid_collisions = False

        future = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self.pr_node, future, timeout_sec=1.0)
        res = future.result()

        if res is None:
            self.pr_node.get_logger().error("[FindObj] IK service call failed (no response)")
            return None

        # error_code: SUCCESS=1
        if res.error_code.val != res.error_code.SUCCESS:
            self.pr_node.get_logger().error(f"[FindObj] IK failed, error_code={res.error_code.val}")
            return None

        # JointState から groupの順番に揃えるのが理想だが、まずは返却順で利用する
        js = res.solution.joint_state
        names = list(js.name)
        pos = list(js.position)

        # xarm6 の6関節だけ抽出したい場合（名前に 'joint' を含む想定でフィルタ）
        # ※環境により名前が違うので、まずはログで確認できるようにする
        self.pr_node.get_logger().info(f"[FindObj] IK joint_state names: {names}")

        return pos

    #TODO: 検出結果を描画して可視化する（全検出を描く）
    def visual_yolo_result(self, r0) -> None:
        """
        r0: ultralytics の 1フレーム分の結果 (results[0])
        - r0.boxes に入っている全てのBBoxを描画して表示する
        """
        if self.cv2_image is None:
            self.pr_node.get_logger().warn("[FindObj] visual_yolo_result: no image")
            return

        # 描画用にコピー（元画像を壊さない）
        vis = self.cv2_image.copy()

        # クラス名辞書
        names = getattr(self.model, "names", {}) if self.model is not None else {}

        if r0 is None or getattr(r0, "boxes", None) is None or len(r0.boxes) == 0:
            cv2.putText(
                vis,
                "No detection",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            for b in r0.boxes:
                # conf/cls/label
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                cls = int(b.cls.item()) if b.cls is not None else -1
                label = str(names.get(cls, str(cls)))

                # xyxy
                xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                x1, y1, x2, y2 = [int(v) for v in xyxy]

                # bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ラベル文字列
                text = f"{label} {conf:.2f}"

                # 文字背景（読みやすくする）
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_text_top = max(0, y1 - th - baseline - 6)
                cv2.rectangle(vis, (x1, y_text_top), (x1 + tw + 6, y1), (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    text,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            self.yolo_pub.publish(img_msg)
        except CvBridgeError as e:
            self.pr_node.get_logger().warn(f"[FindObj] CvBridgeError in visual_yolo_result: {e}")

    def _detect_workpiece(self) -> Optional[Tuple[int, int, int, int, float, str]]:
        """
        画像から self.workpiece を検出して一番それっぽいBBoxを返す
        return: (x1, y1, x2, y2, conf, label)  or None
        """
        if self.model is None or self.cv2_image is None:
            self.pr_node.get_logger().warn(
                f"[FindObj] not ready: model_none={self.model is None}, img_none={self.cv2_image is None}"
            )
            return None

        try:
            # ultralytics は BGR でもだいたい動くが、気になるなら RGB に変換
            img = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)

            results = self.model.predict(
                source=img,
                conf=self.conf_th,
                iou=self.iou_th,
                verbose=False,
            )
            if not results:
                self.pr_node.get_logger().error("cannot predict by yolo, results is None")
                return None

            r0 = results[0] # 0番目が検出結果
            # 可視化
            self.visual_yolo_result(r0)
            if r0.boxes is None or len(r0.boxes) == 0:
                self.pr_node.get_logger().error("results is None")
                return None


            best = None
            best_score = -1.0

            # クラス名辞書（model.names）
            names = getattr(self.model, "names", {})

            for b in r0.boxes:
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                cls = int(b.cls.item()) if b.cls is not None else -1
                label = str(names.get(cls, str(cls)))

                # workpiece が空なら「一番conf高いもの」
                # workpiece が指定されているなら、ラベル一致（部分一致）を優先
                if self.workpiece:
                    if self.workpiece not in label:
                        continue

                xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                x1, y1, x2, y2 = [int(v) for v in xyxy]

                if conf > best_score:
                    best_score = conf
                    best = (x1, y1, x2, y2, conf, label)

            # workpiece 指定で見つからなかった場合、フォールバックで最大confを返す（必要なら）
            if best is None and not self.workpiece:
                return None

            return best

        except Exception as e:
            self.pr_node.get_logger().warn(f"[FindObj] YOLO inference error: {e}")
            return None

    def set_camera_model(self, camera_info) -> None:
        try:
            if self.camera_model is None:
                self.camera_model = PinholeCameraModel()
        except Exception as e:
            self.pr_node.get_logger().warn(f"[FindObj] PinholeCameraModel could not be initialized: {e}")
            return
        self.camera_model.fromCameraInfo(camera_info)
        self.camera_frame_id = camera_info.header.frame_id

    def get_3d_poses(self, depth, bboxes):
        """カメラ座標系での三次元座標（物体中心）を取得します．

        Args:
            depth (np.ndarray): Depth画像．
            bboxes (Tensor): BBox情報．

        Returns:
            List[Optional[Pose]]: 各オブジェクトの3次元座標．
                計算できなかった場合，Noneが格納される．
        """
        poses: List[Optional[Pose]] = []
        for box in bboxes:
            w, h = box[2] - box[0], box[3] - box[1]
            cx, cy = int(box[0] + w / 2), int(box[1] + h / 2)
            crop_depth = depth[cy - 2 : cy + 3, cx - 2 : cx + 3] * 0.001
            flat_depth = crop_depth[crop_depth != 0].flatten()
            if len(flat_depth) == 0:
                poses.append(None)
                continue
            mean_depth = np.mean(flat_depth)
            uv = list(self.camera_model.projectPixelTo3dRay((cx, cy)))
            uv[:] = [x / uv[2] for x in uv]
            uv[:] = [x * mean_depth for x in uv]
            if self._p_max_distance < 0 or (self._p_max_distance > 0 and self._p_max_distance > uv[2]):
                poses.append(
                    Pose(
                        position=Point(x=uv[0], y=uv[1], z=uv[2]),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                    )
                )
            else:
                poses.append(None)
        return poses

    def execute(self, blackboard: Any = None) -> str:
        self.pr_node.get_logger().info("------------------------------------------------")
        self.pr_node.get_logger().info("FindObj state executed.")
        self.pr_node.get_logger().info("------------------------------------------------")
        def _bb_get(key, default=None):
            try:
                value = blackboard.get(key)
                return value if value is not None else default
            except Exception:
                return getattr(blackboard, key, default)

        def _bb_set(key, value):
            try:
                blackboard[key] = value
            except Exception:
                setattr(blackboard, key, value)

        resolved_workpiece = _bb_get("workpiece", None) or self.workpiece

        if resolved_workpiece:
            self.pr_node.get_logger().info(
                f"[FindObj] workpiece already resolved: {resolved_workpiece}. Skip active detection."
            )

            dummy_obj_joints = [0.916, 0.724, -1.70014, 0.001, 0.977, -0.67]

            _bb_set("workpiece", resolved_workpiece)
            if not _bb_get("obj_joints", None):
                _bb_set("obj_joints", dummy_obj_joints)

            return "success"

        try:
            self.xarm.gripper_open()
        except Exception as e:
            self.pr_node.get_logger().warn(f"[FindObj] gripper_open failed but continue: {e}")

        target = None

        # 画像/深度/CameraInfoが揃うまで少し待つ
        t0 = time.time()
        timeout = 2.0  # 秒（必要なら伸ばす）
        while rclpy.ok():
            ok_img = (self.cv2_image is not None)
            ok_dep = (self.cv2_depth is not None)
            ok_info = (self.camera_frame_id != "")
            if ok_img and ok_dep and ok_info:
                break
            if time.time() - t0 > timeout:
                self.pr_node.get_logger().warn(
                    f"[FindObj] wait timeout: img={ok_img}, depth={ok_dep}, info={ok_info}"
                )
                self.pr_node.get_logger().warn(
                    "[FindObj] No detection yet. Skipping detection and returning dummy pose (Simulator mode)."
                )
                obj_joints = [0.916, 0.724, -1.70014, 0.001, 0.977, -0.67]
                setattr(blackboard, "obj_joints", obj_joints)
                return "success"
            rclpy.spin_once(self.pr_node, timeout_sec=0.1)

        # TODO: ここに認識処理を実装していく
        # self.workpieceを認識してobj_jointsを取得する
        # YOLO26
        # 検出
        det = self._detect_workpiece()
        if det is None:
            self.pr_node.get_logger().info("[FindObj] no detection yet -> loop")
            return "loop"

        x1, y1, x2, y2, conf, label = det
        self.pr_node.get_logger().info(
            f"[FindObj] detected: label={label}, conf={conf:.2f}, bbox=({x1},{y1})-({x2},{y2})"
        )

        #TODO: 画像位置 → 座標（pixel → mm）
        # depth画像が必要
        
        if self.cv2_depth is None:
            self.pr_node.get_logger().error("[FindObj] no depth image")
            return "except"
        # BBox中心のDepth値を取得
        # BBox中心（color画像座標）
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # depth画像が必要
        if self.cv2_depth is None or self.cv2_image is None:
            self.pr_node.get_logger().error("[FindObj] no depth or color image")
            return "except"

        # --- ここが修正：color座標 -> depth座標へ変換してから読む ---
        dh, dw = self.cv2_depth.shape[:2]
        ch, cw = self.cv2_image.shape[:2]

        # いまの状況確認ログ（任意だけど超おすすめ）
        self.pr_node.get_logger().info(
            f"[FindObj] color size (w,h)=({cw},{ch}), depth size (w,h)=({dw},{dh}), "
            f"bbox center(color)=({cx},{cy})"
        )

        # スケール変換（解像度が違う前提）
        sx = dw / float(cw)
        sy = dh / float(ch)
        dcx = int(round(cx * sx))
        dcy = int(round(cy * sy))

        # 念のためクリップ（範囲外アクセス防止）
        dcx = max(0, min(dw - 1, dcx))
        dcy = max(0, min(dh - 1, dcy))

        depth_value = self.cv2_depth[dcy, dcx]

        if depth_value == 0:
            self.pr_node.get_logger().error(
                f"[FindObj] depth value is 0 at depth-pixel=({dcx},{dcy}) mapped from color-pixel=({cx},{cy})"
            )
            return "except"

        self.pr_node.get_logger().info(
            f"[FindObj] depth at center: {depth_value} (raw) "
            f"depth-pixel=({dcx},{dcy}) from color-pixel=({cx},{cy})"
        )
        # --- 修正ここまで ---

        # カメラの内部パラメータを使って3D座標を計算
        if not self.camera_model:
            self.pr_node.get_logger().error("[FindObj] camera model not initialized")
            return "except"
        # --- camera内部パラメータ取得（PinholeCameraModelはメソッド） ---
                # --- camera_model 未初期化ガード（CameraInfo未受信だとPがNoneで落ちる） ---
        if self.camera_model is None or getattr(self.camera_model, "P", None) is None:
            self.pr_node.get_logger().error(
                "[FindObj] camera_model not initialized yet (CameraInfo not received)."
            )
            return "except"

        # --- PinholeCameraModel は値がメソッドなので fx()/cx() の形で取得 ---
        fx = self.camera_model.fx()
        fy = self.camera_model.fy()
        cx0 = self.camera_model.cx()
        cy0 = self.camera_model.cy()

        # depth-pixel (dcx, dcy) を使って正規化（今の実装に合わせる）
        x_normalized = (dcx - cx0) / fx
        y_normalized = (dcy - cy0) / fy

        z = float(depth_value) * 0.001  # mm -> m（depthがmmの場合）
        x = x_normalized * z
        y = y_normalized * z

        self.pr_node.get_logger().info(
            f"[FindObj] 3D position in camera frame: x={x:.3f} m, y={y:.3f} m, z={z:.3f} m"
        )


        #TODO: camera座標系 → hand座標系に変換


        #TODO: hand座標系（T空間）から逆運動学（IK)を解いてC空間へ
        # -----------------------
        # camera座標系 -> base座標系へ変換 (TF)
        # -----------------------
        # いま持っている (x,y,z) は camera frame の位置[m]
        # ここでは「マウス中心点」を base frame に変換する
        base_xyz = self._camera_point_to_base(x, y, z)
        if base_xyz is None:
            return "except"

        bx, by, bz = base_xyz
        self.pr_node.get_logger().info(
            f"[FindObj] {self.workpiece} position in base frame: x={bx:.3f} m, y={by:.3f} m, z={bz:.3f} m"
        )
        bz = -0.032 #TODO: 仮の値（テーブル高さ）

        # -----------------------
        # base座標系の目標Poseを作る（姿勢は仮：手先を下向きに）
        # ※ここはタスクに合わせて姿勢とオフセットを調整する
        # -----------------------
        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.header.stamp = self.pr_node.get_clock().now().to_msg()

        # 位置：そのまま掴みに行くと危ないので、少し上にオフセット（例: +5cm）
        self.approach_offset_x = 0.0
        self.approach_offset_y = 0.04
        self.approach_offset_z = -0.03
        self.finger_length = 0.17
        target.pose.position.x = float(bx + self.approach_offset_x)
        target.pose.position.y = float(by + self.approach_offset_y)
        target.pose.position.z = float(bz + self.approach_offset_z + self.finger_length)

                # 姿勢：現在のEE姿勢をコピー（IKが通りやすい）
        cur = self._get_current_ee_pose_in_base()
        if cur is None:
            self.pr_node.get_logger().error("[FindObj] cannot get current ee pose -> except")
            return "except"
        target.pose.orientation = cur.pose.orientation

        # -----------------------
        # IK (Pose -> Joint angles)
        # -----------------------
        obj_joints_ik = self._compute_ik(target)
        if obj_joints_ik is None:
            self.pr_node.get_logger().error("[FindObj] cannot compute IK -> except")
            return "except"

        # ここで6軸分だけ使いたいなら先頭6つを採用（返ってくる順序が環境依存のため暫定）
        obj_joints = obj_joints_ik[:6]
        self.pr_node.get_logger().info(f"[FindObj] obj_joints(from IK) = {obj_joints}")


        # Blackboardに認識結果をセット
        self.pr_node.get_logger().info("[FindObj] set obj_joints")
        setattr(blackboard, "obj_joints", obj_joints) #TODO 一旦C空間のままセット
        return "success"
