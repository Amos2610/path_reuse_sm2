#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
from yasmin.state import State
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper, XArmRobotUtils
from path_reuse_sm2.core.path_registry import PathRegistry
from path_reuse_sm2.core.grasp_orientation import approach_from_orientation, rotate_about_approach
from path_reuse_sm2.core import plan_helpers as ph
from path_reuse_sm2.core import trial_record as tr
from path_reuse_sm2.core.object_layer import attach_object


class Grasp(XArmUtilsWrapper, State):
    def __init__(self, node, **kwargs):
        State.__init__(self, outcomes=["success", "loop", "except"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.pr_client = None  # lazy init via _ensure_path_seed_client()
        self.phase = kwargs.get("phase", "")
        self.workpiece = kwargs.get("workpiece", "")
        self.path_seed_path = kwargs.get("path_seed_path", "")
        self.obj_joints = kwargs.get("joints", [])
        self.grasp_pose = kwargs.get("grasp_pose") or kwargs.get("pose")
        self.robot_utils = XArmRobotUtils(
            self.pr_node,
            base_frame=kwargs.get("base_frame", "link_base"),
            ee_link=kwargs.get("ee_link", "link_tcp"),
            move_group=kwargs.get("move_group", "xarm6"),
            ik_service=kwargs.get("ik_service", "/compute_ik"),
            xarm=self.xarm,
            default_pose_frame=(
                kwargs.get("grasp_pose_frame_id")
                or kwargs.get("pose_frame_id")
                or "camera_color_optical_frame"
            ),
        )
        self.pr_node.get_logger().info(
            f"[Grasp][frame] init base_frame={repr(self.robot_utils.base_frame)}, "
            f"default_pose_frame={repr(self.robot_utils.default_pose_frame)}, "
            f"kwargs.grasp_pose_frame_id={repr(kwargs.get('grasp_pose_frame_id', ''))}, "
            f"kwargs.pose_frame_id={repr(kwargs.get('pose_frame_id', ''))}, "
            f"grasp_pose_type={type(self.grasp_pose).__name__}, "
            f"grasp_pose_len={len(self.grasp_pose) if isinstance(self.grasp_pose, (list, tuple)) else 'n/a'}"
        )

        # Path Registry
        self.source_id = kwargs.get("source_location", "HOME")
        # Grasp は (workpiece の場所, workpiece_id) の組み合わせで管理
        self.target_id = kwargs.get("workpiece") or kwargs.get("target_location") or ""
        self.skill_name = kwargs.get("skill_name", "SkillGraspObj")
        self.step_index = kwargs.get("step_index", 0)
        
        ws_root = self._get_workspace_root()
        registry_path = self.pr_node.get_parameter("pathseed_registry_path").value
        if not registry_path.startswith('/'):
            registry_path = os.path.join(ws_root, registry_path)
            
        self.registry = PathRegistry(registry_path)

        self.pre_grasp_offset = kwargs.get("pre_grasp_offset", 0.1)  # meters
        # link_base座標系での固定orientation [x,y,z,w]。
        # デフォルト: [1,0,0,0] = X軸まわり180° = このロボットの下向き標準姿勢
        _down = [1.0, 0.0, 0.0, 0.0]
        self.pre_grasp_orientation = kwargs.get("pre_grasp_orientation", _down)
        self.grasp_orientation = kwargs.get("grasp_orientation", _down)
        self._down_orientation = list(_down)
        # 接近方向（ベースフレーム）。姿勢から都度導くので None のまま
        self._approach_base = None
        self._tf_broadcaster = TransformBroadcaster(self.pr_node)

        # variables
        self.try_count: int = 0
        self.pipeline: str = "stomp"  # "stomp" or "ompl"
        self.params: Dict[str, Any] = {}  # ROS1の~Params代替。必要ならbb/ファイルから供給

        # STOMP/OMPL共通のデフォルト（必要に応じて上書き）
        self.max_retries_default = 2
        self.max_velocity_scale_default = 0.3
        self.max_accel_scale_default = 0.3
        self.planning_time_default = float(ph.param(self.pr_node, "planning_time_grasp", 5.0))

    def _grasp_done(self, simulate_only: bool) -> Optional[str]:
        """把持後に物体を planning scene へ attach する。None なら成功、文字列なら outcome。

        simulate_only でも呼ぶ（planning scene の操作のみ。続く Put の計画が「持っている
        状態」で行われる。simulate の後は sm_node が巻き戻す）。
        """
        object_id = (self.workpiece or "").strip() or "prsm_workpiece"
        bb = self._current_blackboard
        def _bb(key):
            if bb is None:
                return None
            try:
                return bb[key] if key in bb else None
            except Exception:
                return getattr(bb, key, None)
        attached = attach_object(
            self.pr_node,
            object_id,
            primitive=_bb("obj_primitive"),
            pose=_bb("obj_pose"),
            frame_id=str(_bb("obj_pose_frame_id") or ""),
            mesh=_bb("obj_mesh"),
            tag="Grasp",
        )
        tr.record_event(self.pr_node, "attach", skill="Grasp", object_id=object_id, ok=bool(attached),
                        simulate_only=bool(simulate_only))
        if not attached:
            self.pr_node.get_logger().error("[Grasp] attach failed. Aborting skill.")
            tr.record_stop(self.pr_node, "E-col", "attach", f"Grasp: attach {object_id} failed")
            return "except"
        if bb is not None:
            try:
                bb["attached_object_id"] = object_id
            except Exception:
                setattr(bb, "attached_object_id", object_id)
        # Blackboard は TaskSet ごとに新規なので、別タスクとして来る Put が参照できるよう
        # ノードにも持たせる（simulate の巻き戻しでは触らない）
        if not simulate_only:
            self.pr_node._held_object_id = object_id
        return None

    def _ensure_path_seed_client(self) -> PathSeedClient:
        if self.pr_client is None:
            self.pr_node.get_logger().info("[Grasp] Creating PathSeedClient.")
            self.pr_client = PathSeedClient()
        return self.pr_client

    def _get_workspace_root(self) -> str:
        import os
        try:
            from ament_index_python.packages import get_package_prefix
            install_prefix = get_package_prefix('path_reuse_sm2')
            return os.path.dirname(os.path.dirname(install_prefix))
        except Exception:
            ament_prefix = os.environ.get('AMENT_PREFIX_PATH', '')
            if ament_prefix:
                first_path = ament_prefix.split(':')[0]
                return os.path.dirname(os.path.dirname(first_path))
            return os.getcwd()

    def _param_list(self, name, default):
        """ノードのパラメータ（数値の配列）を読む。未宣言なら default。"""
        try:
            v = self.pr_node.get_parameter(name).value
        except Exception:
            return list(default)
        if v is None:
            return list(default)
        try:
            return [float(x) for x in v]
        except TypeError:
            return [float(v)]

    def _yaw_search_deg(self):
        """接近軸まわりに振る角度[deg]の列（パラメータ grasp_yaw_search_deg）。

        平行グリッパでは 180 度以外は把持方向が変わるので、既定は 180 度だけ。
        対称な物体で広げるときはパラメータで指定する。
        """
        return [d for d in self._param_list("grasp_yaw_search_deg", [180.0]) if abs(d) > 1e-6]

    def _shift_search_m(self):
        """把持位置を水平にずらす量[m]の列（パラメータ grasp_shift_search_m）。

        物体の形状情報が無いので範囲の制約は掛けられず、平行グリッパでは把持点を
        ずらすと掴めなくなる。既定は空（無効）。
        """
        return [d for d in self._param_list("grasp_shift_search_m", [0.0]) if d > 1e-6]

    def _offset_pose_along_approach(self, pose_stamped, offset_m: float):
        """接近方向の手前 offset_m に pre-grasp を置く（ベースフレーム）。

        link_base の +Z に足すだけだと、姿勢を斜めにしたとき pre-grasp が接近直線から
        外れる。いま使う姿勢から接近方向（TCP 局所 +Z）を取り、その逆へ下がる。
        真下固定 [1, 0, 0, 0] では接近方向が (0, 0, -1) なので +Z へ offset_m 上がる。
        self.pre_grasp_orientation が設定されている場合はそのorientationで上書きする。
        """
        pre_grasp = deepcopy(pose_stamped)
        approach = (
            approach_from_orientation(self.pre_grasp_orientation)
            if self.pre_grasp_orientation is not None
            else None
        )
        if approach is None:
            # 姿勢が無い・壊れているときだけ、旧来の真上固定に落とす。
            pre_grasp.pose.position.z += offset_m
        else:
            pre_grasp.pose.position.x -= approach[0] * offset_m
            pre_grasp.pose.position.y -= approach[1] * offset_m
            pre_grasp.pose.position.z -= approach[2] * offset_m
        if self.pre_grasp_orientation is not None:
            q = self.pre_grasp_orientation
            pre_grasp.pose.orientation.x = float(q[0])
            pre_grasp.pose.orientation.y = float(q[1])
            pre_grasp.pose.orientation.z = float(q[2])
            pre_grasp.pose.orientation.w = float(q[3])
        return pre_grasp

    def _rescue_pre_grasp(self, grasp_pose_base, seed_joints):
        """pre-grasp の IK が解けないときに、接近軸まわりに回して解ける姿勢を探す。

        接近方向は変えず、腕の姿勢だけ別の解に移す。採用した姿勢は
        self.grasp_orientation にも入れ、直後の把持目標の解き直しと揃える。
        戻り値: (pre_grasp_pose_stamped, pre_grasp_joints)。全滅なら None。
        """
        original = list(self.pre_grasp_orientation) if self.pre_grasp_orientation is not None else None
        if original is None:
            return None
        approach = approach_from_orientation(original)
        if approach is None:
            return None

        def _try(orientation, label):
            self.pre_grasp_orientation = orientation
            pose = self._offset_pose_along_approach(grasp_pose_base, self.pre_grasp_offset)
            joints = self.robot_utils.normalize_joint_list(
                self.robot_utils.compute_ik(pose, seed_joints=seed_joints)
            )
            if joints is None:
                return None
            self.grasp_orientation = orientation
            self.pr_node.get_logger().warn(
                f"[Grasp] 元の姿勢では pre-grasp の IK が解けないので{label}。"
                f"quat={[round(v, 4) for v in orientation]}"
            )
            return pose, joints

        degs = self._yaw_search_deg()
        for deg in degs:
            rotated = rotate_about_approach(original, approach, math.radians(deg))
            if rotated is None:
                break
            found = _try(rotated, f"接近軸まわりに {deg:g} 度回した")
            if found is not None:
                return found
        self.pr_node.get_logger().warn(
            f"[Grasp] 接近軸まわりに {len(degs)} 通り振ったがどれも pre-grasp の IK が解けなかった"
        )
        self.pre_grasp_orientation = original
        return None

    def _solve_grasp_goal_collision_free(self, grasp_pose_oriented, pre_grasp_joints):
        """把持目標を**干渉しない**姿勢で解く。

        目標そのものが planning scene と干渉していると STOMP は何回やっても通らない。
        順に試す: 1. そのまま  2. 接近軸まわりの yaw（grasp_yaw_search_deg）
        3. 把持位置を水平にずらす（grasp_shift_search_m、既定は無効）。
        戻り値: (goal_joints, (pre_grasp_pose, pre_grasp_joints) or None, shifted_grasp_pose or None)。
        全部干渉するなら (None, None, None) で、呼び手が干渉解に落とす。
        """
        o = grasp_pose_oriented.pose.orientation
        base_q = [o.x, o.y, o.z, o.w]
        approach = approach_from_orientation(base_q)  # 真下把持なら (0, 0, -1)

        orientations = [(0.0, base_q)]
        if approach is not None:
            for deg in self._yaw_search_deg():
                q = rotate_about_approach(base_q, approach, math.radians(deg))
                if q is None:
                    break
                orientations.append((deg, q))

        def _ik(pose):
            return self.robot_utils.normalize_joint_list(
                self.robot_utils.compute_ik(
                    pose, seed_joints=pre_grasp_joints,
                    avoid_collisions=True, fallback_to_collision=False,
                )
            )

        def _with(pose, q, dx, dy):
            p = deepcopy(pose)
            p.pose.position.x += dx
            p.pose.position.y += dy
            p.pose.orientation.x = float(q[0])
            p.pose.orientation.y = float(q[1])
            p.pose.orientation.z = float(q[2])
            p.pose.orientation.w = float(q[3])
            return p

        # 1, 2: 位置はそのまま、姿勢だけ
        for deg, q in orientations:
            joints = _ik(_with(grasp_pose_oriented, q, 0.0, 0.0))
            if joints is None:
                continue
            if deg == 0.0:
                return joints, None, None
            self.pr_node.get_logger().warn(
                f"[Grasp] 把持目標が干渉するので接近軸まわりに {deg:g} 度回した。"
                f"quat={[round(v, 4) for v in q]}"
            )
            return joints, self._realign_pre_grasp(grasp_pose_oriented, q, joints), None

        # 3: 水平にずらす（パラメータで有効化したときだけ）
        shifts = []
        for d in self._shift_search_m():
            shifts.extend([(d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)])
        for dx, dy in shifts:
            for deg, q in orientations:
                pose = _with(grasp_pose_oriented, q, dx, dy)
                joints = _ik(pose)
                if joints is None:
                    continue
                self.pr_node.get_logger().warn(
                    f"[Grasp] 把持目標が干渉するので把持位置を base で ({dx:+.2f}, {dy:+.2f}) m ずらした"
                    + (f"（yaw {deg:g} 度）" if deg else "")
                )
                return joints, self._realign_pre_grasp(pose, q, joints), pose

        self.pr_node.get_logger().warn(
            f"[Grasp] 把持目標は yaw {len(orientations)} 通り・ずらし {len(shifts)} 通りのどれでも干渉する。干渉解で続ける"
        )
        tr.record_event(self.pr_node, "ik_search_exhausted", skill="Grasp", yaw=len(orientations), shift=len(shifts))
        return None, None, None

    def _realign_pre_grasp(self, grasp_pose, q, goal_joints):
        """採用した把持姿勢に pre-grasp を揃える。解けなければ None（呼び手は元のまま）。"""
        self.grasp_orientation = list(q)
        self.pre_grasp_orientation = list(q)
        pre_pose = self._offset_pose_along_approach(deepcopy(grasp_pose), self.pre_grasp_offset)
        pre_joints = self.robot_utils.normalize_joint_list(
            self.robot_utils.compute_ik(pre_pose, seed_joints=goal_joints)
        )
        if pre_joints is None:
            self.pr_node.get_logger().warn(
                "[Grasp] 直した姿勢では pre-grasp の IK が解けない。pre-grasp は元のまま"
            )
            return None
        return pre_pose, pre_joints

    def _publish_grasp_tfs(self, grasp_ps, pre_grasp_ps=None):
        """grasp, pre_graspをTFフレームとして配信する（RViz可視化用）。"""
        now = self.pr_node.get_clock().now().to_msg()
        frames = [("grasp_target", grasp_ps)]
        if pre_grasp_ps is not None:
            frames.append(("pre_grasp_target", pre_grasp_ps))
        for child_frame_id, ps in frames:
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = self.robot_utils.base_frame  # 常に link_base 基準
            t.child_frame_id = child_frame_id
            t.transform.translation.x = ps.pose.position.x
            t.transform.translation.y = ps.pose.position.y
            t.transform.translation.z = ps.pose.position.z
            t.transform.rotation = ps.pose.orientation
            self._tf_broadcaster.sendTransform(t)
        self.pr_node.get_logger().info(
            f"[Grasp] Published TF: {[f for f, _ in frames]} in {self.robot_utils.base_frame}"
        )

    # ====== ROS1: set_stomp_params ======
    # 目的: 与えられたphase(dict)をMoveItへ適用（ROS2ではrosparamでなくAPI直適用）
    def set_stomp_params(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set STOMP parameters into MoveIt via XArmUtils.set_move_group_parameter
        Args:
            phase: フェーズごとのSTOMPパラメータ辞書（ROS1互換）
        Returns:
            適用したパラメータ（検証ログ用に返す）
        """
        # 代表的なキー例。存在するものだけ適用（ROS1 -> ROS2移行の最小公倍数）
        mapping = {
            # 速度・加速度・計画時間（STOMP/OMPL共通でも使える）
            "planning_time": "planning_time",
            "max_velocity_scaling_factor": "max_velocity_scaling_factor",
            "max_acceleration_scaling_factor": "max_acceleration_scaling_factor",

            # 以下はSTOMP拡張（MoveIt側のパラメータ名に合わせて適宜調整）
            # 必要になったらキー名をあなたの環境に合わせて追加
            "num_iterations": "stomp/num_iterations",
            "num_iterations_after_valid": "stomp/num_iterations_after_valid",
            "num_timesteps": "stomp/num_timesteps",
            "delayed_collision_checking": "stomp/delayed_collision_checking",
            "update_multiplier": "stomp/update_multiplier",
        }

        for k_src, k_dst in mapping.items():
            if k_src in phase:
                v = phase[k_src]
                # 代表的な型に合わせて設定（bool/int/float/str対応）
                if isinstance(v, bool):
                    self.xarm.set_move_group_parameter(k_dst, bool(v))
                elif isinstance(v, int):
                    self.xarm.set_move_group_parameter(k_dst, int(v))
                elif isinstance(v, float):
                    self.xarm.set_move_group_parameter(k_dst, float(v))
                elif isinstance(v, str):
                    self.xarm.set_move_group_parameter(k_dst, str(v))
        return phase
    
    def _resolve_pathseed_file(self, pathseed_file: str) -> str:
        """
        pathseedの相対パスをPathSeedLibrary基準で絶対パス化する．
        """
        if not pathseed_file:
            return ""

        legacy_map = {
            "ex1_pick_and_place/pre_defined/pathseed_pick.txt": "ex1_pick_and_place/pre_defined/pick/pathseed_pick_0529_STOMP01.txt",
            "ex1_pick_and_place/pre_defined/pathseed_place.txt": "ex1_pick_and_place/pre_defined/place/pathseed_place_0627_RRT-connect01.txt",
        }
        pathseed_file = legacy_map.get(pathseed_file, pathseed_file)

        if pathseed_file.startswith("/"):
            return pathseed_file

        ws_root = self._get_workspace_root()

        if pathseed_file.startswith("src/"):
            return os.path.join(ws_root, pathseed_file)

        library_base = os.path.join(
            ws_root,
            "src/path_reuse_method/pathseeds/Library",
        )
        return os.path.join(library_base, pathseed_file)

    def generate_stomp_path_from_pathseed(
        self,
        file_path: str,
        start_joint_values: List[float],
        goal_joint_values: List[float],
    ) -> bool:
        """
        PathSeed ファイルを decode → set する（ROS2版）
        - ROS1のDecoder+param set の代わりに PathSeedClient を使用
        """
        try:
            self.pr_node.get_logger().info("[Grasp] Decoding pathseed via PathSeedClient...")
            pr_client = self._ensure_path_seed_client()
            decoded_path = pr_client.send_decode_path_seed(file_path, start_joint_values, goal_joint_values)
            rows = int(getattr(decoded_path, "rows", 0) or 0) if decoded_path is not None else 0
            tr.record_event(self.pr_node, "seed", skill="Grasp", path=str(file_path), rows=rows)
            if decoded_path is None or rows == 0:
                self.pr_node.get_logger().error(f"[Grasp] decode failed: {'None returned' if decoded_path is None else 'empty seed (rows=0)'}: {file_path}")
                tr.record_stop(self.pr_node, "E-seed", "seed_decode", f"{file_path}")
                return False
            self.pr_node.get_logger().info("[Grasp] Setting decoded pathseed...")
            pr_client.send_set_path_seed(decoded_path)

            # デバッグ用に残したい場合はBBへ保存する運用に
            # bb.pathseed_decoded = decoded_path

            return True
        except Exception as e:
            self.pr_node.get_logger().error(f"[Grasp] Exception in generate_stomp_path_from_pathseed: {e}")
            return False
    
    def set_start_and_goal_joint_values(self, blackboard: Any) -> Optional[Tuple[List[float], List[float]]]:
        def _bb_get(key, default=None):
            try:
                value = blackboard.get(key)
                return value if value is not None else default
            except Exception:
                try:
                    return getattr(blackboard, key)
                except Exception:
                    return default

        raw_obj_joints = _bb_get("obj_joints", None)
        obj_joints = self.robot_utils.resolve_joints(
            raw_obj_joints,
            self.obj_joints,
            self.grasp_pose,
        )

        if not obj_joints:
            self.pr_node.get_logger().error("Blackboard missing valid 'obj_joints' and IK fallback failed.")
            return None

        try:
            blackboard["obj_joints"] = obj_joints
        except Exception:
            try:
                setattr(blackboard, "obj_joints", obj_joints)
            except Exception:
                pass

        move_joints = _bb_get("move_joints", None)

        if not self.robot_utils.is_joint_list(move_joints):
            current_joints = self.robot_utils.get_current_joint_values()
            if current_joints:
                self.pr_node.get_logger().info(
                    "Blackboard missing 'move_joints'. Use current joint values."
                )
                move_joints = current_joints
            else:
                # TODO(fake): when current joints are unavailable in fake mode,
                # use a defined safe start posture. For real robot, this branch should rarely be used.
                self.pr_node.get_logger().warn(
                    "Blackboard missing 'move_joints' and current joints are unavailable. Use fallback start joints for simulator mode."
                )
                move_joints = [0.916, 0.724, -1.70014, 0.001, 0.977, -0.67]

            try:
                blackboard["move_joints"] = move_joints
            except Exception:
                try:
                    setattr(blackboard, "move_joints", move_joints)
                except Exception:
                    pass

        start_joint_values = move_joints
        goal_joint_values = obj_joints

        self.pr_node.get_logger().info(f"Grasp start joint values: {start_joint_values}")
        self.pr_node.get_logger().info(f"Grasp goal joint values: {goal_joint_values}")

        return start_joint_values, goal_joint_values

    def _publish_display_trajectory(self, plan_jt) -> None:
        """simulate_only 時に軌道をループパブリッシャーに渡す。"""
        if not hasattr(self.pr_node, '_viz_traj_loop'):
            return
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = plan_jt
        disp = DisplayTrajectory()
        disp.model_id = "UF_ROBOT"
        disp.trajectory.append(robot_traj)
        interval = 5.0
        if plan_jt.points:
            last = plan_jt.points[-1].time_from_start
            duration = last.sec + last.nanosec * 1e-9
            interval = max(duration + 1.0, 3.0)
        self.pr_node._viz_traj_loop.start(disp, interval)
        self.pr_node.get_logger().info(
            f"[Grasp] Trajectory loop started (interval={interval:.1f}s, {len(plan_jt.points)} points)."
        )

    def execute(self, blackboard=None):
        self.pr_node.get_logger().info("------------------------------------------------")
        self.pr_node.get_logger().info(f"Grasp state executed.")
        self.pr_node.get_logger().info("------------------------------------------------")
        self._current_blackboard = blackboard

        def _bb_get(key, default=None):
            try:
                return blackboard[key]
            except Exception:
                try:
                    value = blackboard.get(key)
                    return value if value is not None else default
                except Exception:
                    try:
                        return getattr(blackboard, key)
                    except Exception:
                        return default

        bb_pose_frame_id = (
            _bb_get("grasp_pose_frame_id", "")
            or _bb_get("pose_frame_id", "")
            or _bb_get("camera_frame_id", "")
        )
        self.pr_node.get_logger().info(
            f"[Grasp][frame] execute before default update: "
            f"bb.grasp_pose_frame_id={repr(_bb_get('grasp_pose_frame_id', ''))}, "
            f"bb.pose_frame_id={repr(_bb_get('pose_frame_id', ''))}, "
            f"bb.camera_frame_id={repr(_bb_get('camera_frame_id', ''))}, "
            f"current_default_pose_frame={repr(self.robot_utils.default_pose_frame)}"
        )
        if bb_pose_frame_id:
            self.robot_utils.default_pose_frame = bb_pose_frame_id
            self.pr_node.get_logger().info(
                f"[Grasp][frame] default grasp pose frame set from blackboard: {repr(bb_pose_frame_id)}"
            )

        # blackboard の grasp_pose を優先して self.grasp_pose に反映する
        bb_grasp_pose = _bb_get("grasp_pose", None)
        if bb_grasp_pose is not None:
            self.grasp_pose = bb_grasp_pose
            self.pr_node.get_logger().info(
                f"[Grasp] grasp_pose loaded from blackboard: {self.grasp_pose}"
            )

        # 認識結果（FindObjなど）があればそれを優先、なければ RAG/KB からの値を使う
        detected_obj_joints = _bb_get("obj_joints", None)

        if self.robot_utils.is_joint_list(detected_obj_joints):
            self.obj_joints = self.robot_utils.normalize_joint_list(detected_obj_joints)
            self.pr_node.get_logger().info(f"Using dynamically detected obj_joints: {self.obj_joints}")
        elif detected_obj_joints is not None:
            self.pr_node.get_logger().info("Detected obj_joints is pose-like. Defer IK to joint resolution.")
        elif self.robot_utils.is_joint_list(self.obj_joints):
            self.obj_joints = self.robot_utils.normalize_joint_list(self.obj_joints)
            try:
                blackboard["obj_joints"] = self.obj_joints
            except Exception:
                setattr(blackboard, "obj_joints", self.obj_joints)
            self.pr_node.get_logger().info(f"Using KB/RAG provided obj_joints: {self.obj_joints}")
        elif self.grasp_pose is not None:
            self.pr_node.get_logger().info("Using grasp_pose for IK fallback.")
        else:
            self.pr_node.get_logger().warn("No obj_joints or grasp_pose before joint resolution.")
            
        self.phase = self.pr_node.get_parameter("grasp_phase").value
        self.pr_node.get_logger().info(f"Current phase: {self.phase}")

        # grasp_orientation が設定されており、まだ関節値が未確定の場合:
        # ベースフレームへ変換後に orientation を上書きしてから IK を計算し、
        # blackboard に書き込む。これにより set_start_and_goal_joint_values が
        # そのまま使用する。
        if (self.grasp_orientation is not None
                and self.grasp_pose is not None
                and not self.robot_utils.is_joint_list(self.obj_joints)):
            grasp_pose_base_tmp = self.robot_utils.transform_to_base(self.grasp_pose)
            if grasp_pose_base_tmp is not None:
                q = self.grasp_orientation
                grasp_pose_base_tmp.pose.orientation.x = float(q[0])
                grasp_pose_base_tmp.pose.orientation.y = float(q[1])
                grasp_pose_base_tmp.pose.orientation.z = float(q[2])
                grasp_pose_base_tmp.pose.orientation.w = float(q[3])
                forced_joints = self.robot_utils.normalize_joint_list(
                    self.robot_utils.compute_ik(grasp_pose_base_tmp)
                )
                if forced_joints is not None:
                    self.pr_node.get_logger().info(
                        f"[Grasp] Applied grasp_orientation override. IK joints: {forced_joints}"
                    )
                    # blackboard 経由は Yasmin の内部 dict に反映されない場合があるため
                    # self.obj_joints に直接セットして resolve_joints に確実に渡す
                    self.obj_joints = forced_joints
                    try:
                        blackboard["obj_joints"] = forced_joints
                    except Exception:
                        try:
                            setattr(blackboard, "obj_joints", forced_joints)
                        except Exception:
                            pass
                else:
                    self.pr_node.get_logger().warn(
                        "[Grasp] grasp_orientation IK failed, falling back to original grasp_pose."
                    )

        ##############################
        ### Fast path: pre-planned ###
        ##############################
        simulate_only = self.pr_node.get_parameter("prsm_simulate_only").value
        skill_key = f"{self.skill_name}_{self.step_index}"

        if not simulate_only:
            pre_plans = blackboard["pre_planned_trajectories"] if "pre_planned_trajectories" in blackboard else {}
            pre_plan = pre_plans.get(skill_key)
            if pre_plan is not None:
                self.pr_node.get_logger().info(f"[Grasp] Executing pre-planned trajectory for {skill_key}.")
                exec_success = self.xarm.execute_with_plan(pre_plan)
                if not exec_success:
                    self.pr_node.get_logger().error("[Grasp] Pre-planned execution failed.")
                    return "except"
                if self._current_blackboard is not None:
                    setattr(self._current_blackboard, 'grasp_trajectory', deepcopy(pre_plan))
                try:
                    self.xarm.gripper_close()
                except Exception as e:
                    self.pr_node.get_logger().warn(f"[Grasp] gripper_close failed but continue: {e}")
                failed = self._grasp_done(simulate_only)
                if failed is not None:
                    return failed
                return "success"

        ######################################
        ### Normal path: resolve → plan → execute/simulate ###
        ######################################
        joint_values = self.set_start_and_goal_joint_values(blackboard)
        if joint_values is None:
            self.pr_node.get_logger().error("Failed to set start/goal joint values.")
            tr.record_stop(self.pr_node, "E-ik", "ik_goal", "Grasp: start/goal joint values unresolved")
            return "except"

        start_joint_values, goal_joint_values = joint_values
        if not goal_joint_values:
            self.pr_node.get_logger().error("Failed to set joint value target.")
            tr.record_stop(self.pr_node, "E-ik", "ik_goal", "Grasp: goal joint values empty")
            return "except"

        # ==============================
        # Pre-grasp 計算と TF 配信
        # ==============================
        grasp_pose_base = None
        pre_grasp_pose_base = None
        pre_grasp_joints = None

        if self.grasp_pose is not None:
            grasp_pose_base = self.robot_utils.transform_to_base(self.grasp_pose)
            if grasp_pose_base is not None:
                pre_grasp_pose_base = self._offset_pose_along_approach(
                    grasp_pose_base, self.pre_grasp_offset
                )
                # Step1: graspをシードにpre-graspを計算（同じ関節構成ファミリーに誘導）
                raw = self.robot_utils.compute_ik(pre_grasp_pose_base, seed_joints=goal_joint_values)
                pre_grasp_joints = self.robot_utils.normalize_joint_list(raw)
                if pre_grasp_joints is None:
                    rescued = self._rescue_pre_grasp(grasp_pose_base, goal_joint_values)
                    if rescued is not None:
                        pre_grasp_pose_base, pre_grasp_joints = rescued
                if pre_grasp_joints is None:
                    self.pr_node.get_logger().warn(
                        "[Grasp] Pre-grasp IK failed, will skip pre-grasp step."
                    )
                else:
                    self.pr_node.get_logger().info(
                        f"[Grasp] Pre-grasp joints: {pre_grasp_joints}"
                    )
                    # Step2: pre-graspをシードにgraspを再計算して同じ関節構成を保証する
                    # （シードなしIKが別の肘構成を返した場合でも、ここで揃える）
                    grasp_pose_base_oriented = deepcopy(grasp_pose_base)
                    if self.grasp_orientation is not None:
                        q = self.grasp_orientation
                        grasp_pose_base_oriented.pose.orientation.x = float(q[0])
                        grasp_pose_base_oriented.pose.orientation.y = float(q[1])
                        grasp_pose_base_oriented.pose.orientation.z = float(q[2])
                        grasp_pose_base_oriented.pose.orientation.w = float(q[3])
                    refined, rescued_pre, shifted = self._solve_grasp_goal_collision_free(
                        grasp_pose_base_oriented, pre_grasp_joints
                    )
                    if rescued_pre is not None:
                        pre_grasp_pose_base, pre_grasp_joints = rescued_pre
                    if shifted is not None:
                        grasp_pose_base = shifted
                    if refined is None:
                        refined = self.robot_utils.normalize_joint_list(
                            self.robot_utils.compute_ik(
                                grasp_pose_base_oriented, seed_joints=pre_grasp_joints
                            )
                        )
                    if refined is not None:
                        goal_joint_values = refined
                        self.pr_node.get_logger().info(
                            f"[Grasp] Refined goal_joint_values (seeded from pre-grasp): {goal_joint_values}"
                        )

        if grasp_pose_base is not None:
            self._publish_grasp_tfs(grasp_pose_base, pre_grasp_pose_base)

        # simulate_only では実機動作を避けるため、グリッパー操作は行わない。
        if not simulate_only:
            try:
                self.xarm.gripper_open()
            except Exception as e:
                self.pr_node.get_logger().warn(f"[Grasp] gripper_open failed but continue: {e}")

        # use_pathseed = True  # TODO: blackboard等で切り替え可能に
        use_pathseed = self.pr_node.get_parameter("use_pathseed").value
        self.pr_node.get_logger().info(f"use_pathseed: {use_pathseed}")
        if use_pathseed:
            try:
                self.xarm.set_planning_pipeline("stomp")
            except Exception as e:
                self.pr_node.get_logger().warn(f"[Grasp] set_planning_pipeline failed but continue: {e}")
            # PathSeedからSTOMP用軌道をセット
            try:
                self.xarm.set_move_group_parameter("stomp.use_custom_trajectory", True)
            except Exception as e:
                self.pr_node.get_logger().warn(f"[Grasp] set_move_group_parameter failed but continue: {e}")
            self.pr_node.get_logger().info(f"Grasp phase: {self.phase}")
            if self.path_seed_path:
                pathseed_file = self.path_seed_path
            else:
                pathseed_file = self._ensure_path_seed_client().select_best_path_seed(
                    environment_id="desk_scene_v1",
                    skill_name="pick",
                    start_joints=start_joint_values,
                    goal_joints=goal_joint_values,
                )
                if not pathseed_file:
                    if self.phase == "Initial_Phase":
                        pathseed_file = self.pr_node.get_parameter("pathseed_grasp").value
                    elif self.phase in ("Implement_Phase", "Imprementation_Phase"):
                        default_pathseed_file = self.pr_node.get_parameter("pathseed_grasp").value
                        pathseed_file = "/".join(default_pathseed_file.split("/")[:-3]) + "/updated/pathseed_pick.txt"
                    else:
                        self.pr_node.get_logger().error(f"Unknown phase: {self.phase}")
                        return "except"

            pathseed_file = self._resolve_pathseed_file(pathseed_file)
            self.pr_node.get_logger().info(f"[Grasp] Using pathseed file: {pathseed_file}")
            success_generated = self.generate_stomp_path_from_pathseed(
                file_path=pathseed_file,
                start_joint_values=start_joint_values,
                goal_joint_values=goal_joint_values
            )
            if not success_generated:
                self.pr_node.get_logger().error("Failed to generate STOMP path from PathSeed.")
                tr.record_stop(self.pr_node, "E-seed", "seed_decode", f"Grasp: {pathseed_file}")
                return "except"
        else:
            try:
                self.xarm.set_planning_pipeline("ompl")
            except Exception as e:
                self.pr_node.get_logger().warn(f"[Grasp] set_planning_pipeline(ompl) failed but continue: {e}")

        ##############################
        ### Planning and Execution ###
        ##############################

        # --- Step 1: Pre-grasp (OMPL, use_pathseed=False のみ) ---
        # simulate_only では実機動作を避けるため、pre-grasp の実行は行わない。
        if not simulate_only and not use_pathseed and pre_grasp_joints is not None:
            self.pr_node.get_logger().info("[Grasp] Planning to pre-grasp position...")
            ph.apply_planning_time(self.xarm, self.pr_node, self.planning_time_default, "Grasp")
            ph.apply_start_state(self.xarm, self.pr_node, simulate_only, start_joint_values, "Grasp")
            self.xarm.set_joint_value_target(pre_grasp_joints)
            success_pre, _, _, _ = self.xarm.plan()
            if success_pre:
                self.pr_node.get_logger().info("[Grasp] Executing pre-grasp...")
                if not self.xarm.execute():
                    self.pr_node.get_logger().error("[Grasp] Pre-grasp execution failed.")
                    return "except"
                self.pr_node.get_logger().info("[Grasp] Pre-grasp reached.")
                # TF再配信（現在時刻で更新）
                self._publish_grasp_tfs(grasp_pose_base, pre_grasp_pose_base)
            else:
                self.pr_node.get_logger().warn(
                    "[Grasp] Pre-grasp planning failed, attempting direct grasp."
                )
        elif simulate_only and not use_pathseed and pre_grasp_joints is not None:
            self.pr_node.get_logger().info(
                "[Grasp] Simulate only: skip pre-grasp execution and plan direct grasp trajectory."
            )

        # --- Step 2: Grasp ---
        ph.apply_planning_time(self.xarm, self.pr_node, self.planning_time_default, "Grasp")
        ph.apply_start_state(self.xarm, self.pr_node, simulate_only, start_joint_values, "Grasp")
        self.xarm.set_joint_value_target(goal_joint_values)
        success, plan, _plan_sec, _plan_err = self.xarm.plan()
        _err_val = int(getattr(_plan_err, "val", 0) or 0)
        tr.record_event(self.pr_node, "plan", skill="Grasp", success=bool(success), error_code=_err_val,
                        planning_sec=round(float(_plan_sec or 0.0), 3), attempt=self.try_count + 1)
        if success:
            if simulate_only:
                self.pr_node.get_logger().info(f"[Grasp] Storing pre-planned trajectory for {skill_key}.")
                self._publish_display_trajectory(plan)
                self.pr_node._pre_planned_trajectories[skill_key] = plan
                if plan is not None and plan.points:
                    ph.set_sim_end_joints(self._current_blackboard, plan.points[-1].positions)
                if self._current_blackboard is not None:
                    setattr(self._current_blackboard, 'grasp_trajectory', deepcopy(plan))
                    self.pr_node.get_logger().info(
                        f"[Grasp] stored grasp_trajectory to BB (points={len(plan.points)})"
                    )
                failed = self._grasp_done(simulate_only)
                if failed is not None:
                    return failed
                return "success"

            self.pr_node.get_logger().info("Plan found, executing...")
            exec_success = self.xarm.execute()
            if not exec_success:
                self.pr_node.get_logger().error("Execution failed.")
                tr.record_stop(self.pr_node, "EXEC", "execution_failed", "Grasp")
                return "except"
            self.pr_node.get_logger().info("Execution succeeded.")
            # Blackboardに軌道を保存
            if self._current_blackboard is not None:
                setattr(self._current_blackboard, 'grasp_trajectory', deepcopy(plan))
                self.pr_node.get_logger().info(f"[Grasp] stored grasp_trajectory to BB (points={len(plan.points)})")
            # spin周りでエラーが出るため、try-exceptで囲む。
            # TODO: 根本的にはXArmUtils（xarm_utils_py）が内部で xarm_core_node を独自の executor でスピンさせているのが原因と思われるため、将来的にはそちらの改修も検討。
            try:
                self.xarm.gripper_close()
            except Exception as e:
                self.pr_node.get_logger().warn(f"[Grasp] gripper_close failed but continue: {e}")
            failed = self._grasp_done(simulate_only)
            if failed is not None:
                return failed
            return "success"
        else:
            # JACIII安全性評価実験用の修正：put.pyと同じ理由で，try_count/
            # max_retries_defaultの上限チェックが未実装のまま無限リトライ
            # していたバグを修正（詳細はput.pyの同箇所コメント参照）。
            self.try_count += 1
            if self.try_count > self.max_retries_default:
                self.pr_node.get_logger().error(
                    f"No valid plan found after {self.try_count} attempts "
                    f"(max_retries_default={self.max_retries_default}). Aborting Grasp skill."
                )
                self.try_count = 0
                tr.record_stop(self.pr_node, "E-col", "planning",
                               f"Grasp: no valid plan after retries (last MoveIt error_code={_err_val})")
                return "except"
            self.pr_node.get_logger().warn(
                f"No valid plan found, retrying... (attempt {self.try_count}/{self.max_retries_default})"
            )
            return "loop"