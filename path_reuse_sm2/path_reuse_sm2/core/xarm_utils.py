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
import threading


# xArm6 の関節名（joint1..joint6）
XARM6_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _call_service(client, request, timeout_sec: float):
    """サービスを呼び、**ノードを spin せずに**応答を待つ。

    ``rclpy.spin_until_future_complete(node, ...)`` はノードをグローバル executor にも
    登録して回すため、PRSM ノードが既に動いている MultiThreadedExecutor と二重所有に
    なる。ロボアプリ版の実測（2026-08-25）では 2 タスク目以降の ``/prsm_task_set`` が
    コールバックに届かなくなった。future の done コールバックで ``threading.Event`` を
    立てて待てば、応答は既に動いている executor のスレッドで配送され、ノードの
    再登録は起きない。応答が来なければ None。
    （ロボアプリ版 nex10_utils.py:29-61 からの移植）
    """
    future = client.call_async(request)
    done = threading.Event()
    future.add_done_callback(lambda _f: done.set())
    if not done.wait(timeout_sec):
        future.cancel()
        return None
    return future.result()


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

    def _param(self, name, default):
        """ノードのパラメータを読む。未宣言なら default。"""
        try:
            value = self.node.get_parameter(name).value
        except Exception:
            return default
        return default if value is None else value

    def _avoid_collisions_default(self) -> bool:
        """IK に衝突回避を要求するか（パラメータ ik_avoid_collisions、未宣言なら True）。"""
        return bool(self._param("ik_avoid_collisions", True))

    def _base_z_override(self):
        """IK 目標の base 系 z を固定値で上書きするか（パラメータ ik_base_z_override）。

        負の値なら上書きしない。既定 0.125 は研究版が決め打ちしていた値で、
        従来挙動を保つために既定として残している。実験構成では -1.0 にして
        検出／KB の高さをそのまま使う。
        """
        v = self._param("ik_base_z_override", 0.125)
        try:
            v = float(v)
        except Exception:
            return None
        return None if v < 0.0 else v

    def compute_ik(
        self,
        pose,
        seed_joints: Optional[List[float]] = None,
        avoid_collisions: Optional[bool] = None,
        fallback_to_collision: bool = True,
    ):
        """IK を解く。

        ``avoid_collisions`` が真（既定はパラメータ ``ik_avoid_collisions``）なら
        planning scene と干渉しない解だけを受け付ける。干渉しない解が無いとき、
        ``fallback_to_collision=False`` なら None を返し（干渉解に落とさない。呼び手が
        別の姿勢を試すためのモード）、True なら理由を WARN に残してから衝突を許して
        解き直す。
        （ロボアプリ版 nex10_utils.py:263-353 からの移植）
        """
        target = self.pose_to_pose_stamped(pose)
        target = self._transform_pose_to_base(target)
        if target is None:
            return None

        z_override = self._base_z_override()
        if z_override is not None:
            target.pose.position.z = z_override

        if self._ik_cli is None:
            self._ik_cli = self.node.create_client(GetPositionIK, self.ik_service)

        if not self._ik_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().error(f"[XArmRobotUtils] IK service not available: {self.ik_service}")
            return None

        def ask(avoid: bool):
            req = GetPositionIK.Request()
            req.ik_request.group_name = self.move_group
            req.ik_request.ik_link_name = self.ee_link
            req.ik_request.pose_stamped = target
            req.ik_request.timeout = DurationMsg(sec=1, nanosec=0)
            req.ik_request.avoid_collisions = bool(avoid)

            # seed_joints をセットすると姿勢の初期ジョイントがセットできる
            if seed_joints is not None and len(seed_joints) >= 6:
                js = JointState()
                js.name = list(XARM6_JOINT_NAMES)
                js.position = [float(v) for v in seed_joints[:6]]
                rs = RobotState()
                rs.joint_state = js
                # is_diff=False だと move_group が attach 物体を捨てて解く
                # （conversions.cpp の clearAttachedBodies）。掴んだ物込みで判定させる
                rs.is_diff = True
                req.ik_request.robot_state = rs

            return _call_service(self._ik_cli, req, timeout_sec=2.0)

        want_avoid = self._avoid_collisions_default() if avoid_collisions is None else bool(avoid_collisions)

        res = ask(want_avoid)
        if res is None:
            self.node.get_logger().error("[XArmRobotUtils] IK service call failed")
            return None

        if res.error_code.val != res.error_code.SUCCESS and want_avoid and not fallback_to_collision:
            self.node.get_logger().warn(
                f"[XArmRobotUtils] 衝突しない IK 解が無い (error_code={res.error_code.val})。"
                "fallback_to_collision=False なので None を返す"
            )
            return None

        if res.error_code.val != res.error_code.SUCCESS and want_avoid:
            # 衝突しない姿勢が無い。黙って衝突する解に落ちると原因がログから読み取れなく
            # なるので、理由を残してから解き直す。IK が見るシーンとプランナが見るシーンは
            # 同じとは限らないため、この警告が出ても計画は通ることがある。
            self.node.get_logger().warn(
                f"[XArmRobotUtils] 衝突しない IK 解が無い (error_code={res.error_code.val})。"
                "衝突を許して解き直す。返る姿勢は障害物か自分自身と干渉している可能性がある"
            )
            res = ask(False)
            if res is None:
                self.node.get_logger().error("[XArmRobotUtils] IK service call failed")
                return None

        if res.error_code.val != res.error_code.SUCCESS:
            self.node.get_logger().error(f"[XArmRobotUtils] IK failed, error_code={res.error_code.val}")
            return None

        joints = list(res.solution.joint_state.position)[:6]
        self.node.get_logger().info(f"[XArmRobotUtils] IK joints: {joints}")
        return joints

    def _read_fresh_joint_state(self, timeout_sec: float = 1.0):
        """/joint_states を使い捨てノードで 1 通だけ読む。

        move_group interface 経由の現在値は spin 枯渇で古い値を返すことがある
        （ロボアプリ版の実測 2026-08-28: 前タスクの把持姿勢が返り、承認後の実行が
        Invalid Trajectory: start point deviates で止まった）。使い捨てノード＋
        自前 executor なので既存ノードの spin とは衝突しない。
        （ロボアプリ版 nex10_utils.py:355-389 からの移植）
        """
        import os as _os
        import time as _time
        from rclpy.executors import SingleThreadedExecutor as _STE
        name = f"xarm_js_probe_{_os.getpid()}_{_time.monotonic_ns()}"
        try:
            probe = rclpy.create_node(name)
        except Exception:
            return None
        got = {}
        probe.create_subscription(JointState, "/joint_states", lambda m: got.setdefault("msg", m), 10)
        ex = _STE()
        ex.add_node(probe)
        try:
            deadline = _time.monotonic() + timeout_sec
            while "msg" not in got and _time.monotonic() < deadline:
                ex.spin_once(timeout_sec=0.1)
        finally:
            ex.remove_node(probe)
            probe.destroy_node()
        msg = got.get("msg")
        if msg is None:
            return None
        by_name = dict(zip(msg.name, msg.position))
        if not all(k in by_name for k in XARM6_JOINT_NAMES):
            return None
        return [float(by_name[k]) for k in XARM6_JOINT_NAMES]

    def get_current_joint_values(self):
        # TODO(real-robot): replace fallback behavior with actual current joint acquisition.
        # In fake MoveIt, current_state can be empty or timestamp 0, so returning None is expected.
        fresh = self._read_fresh_joint_state()
        if fresh is not None:
            return fresh
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

