#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        self._tf_broadcaster = TransformBroadcaster(self.pr_node)

        # variables
        self.try_count: int = 0
        self.pipeline: str = "stomp"  # "stomp" or "ompl"
        self.params: Dict[str, Any] = {}  # ROS1の~Params代替。必要ならbb/ファイルから供給

        # STOMP/OMPL共通のデフォルト（必要に応じて上書き）
        self.max_retries_default = 2
        self.max_velocity_scale_default = 0.3
        self.max_accel_scale_default = 0.3
        self.planning_time_default = 2.0

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

    def _offset_pose_along_approach(self, pose_stamped, offset_m: float):
        """link_base座標系のZ軸上方にoffset_mだけ離れたpre_graspを計算する（上方向把持用）。
        self.pre_grasp_orientation が設定されている場合はそのorientationで上書きする。
        """
        pre_grasp = deepcopy(pose_stamped)
        pre_grasp.pose.position.z += offset_m
        if self.pre_grasp_orientation is not None:
            q = self.pre_grasp_orientation
            pre_grasp.pose.orientation.x = float(q[0])
            pre_grasp.pose.orientation.y = float(q[1])
            pre_grasp.pose.orientation.z = float(q[2])
            pre_grasp.pose.orientation.w = float(q[3])
        return pre_grasp

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
            if decoded_path is None:
                self.pr_node.get_logger().error("[Grasp] decode failed: None returned.")
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
                return "success"

        ######################################
        ### Normal path: resolve → plan → execute/simulate ###
        ######################################
        joint_values = self.set_start_and_goal_joint_values(blackboard)
        if joint_values is None:
            self.pr_node.get_logger().error("Failed to set start/goal joint values.")
            return "except"

        start_joint_values, goal_joint_values = joint_values
        if not goal_joint_values:
            self.pr_node.get_logger().error("Failed to set joint value target.")
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
        self.xarm.set_joint_value_target(goal_joint_values)
        success, plan, _, _ = self.xarm.plan()
        if success:
            if simulate_only:
                self.pr_node.get_logger().info(f"[Grasp] Storing pre-planned trajectory for {skill_key}.")
                self._publish_display_trajectory(plan)
                self.pr_node._pre_planned_trajectories[skill_key] = plan
                if self._current_blackboard is not None:
                    setattr(self._current_blackboard, 'grasp_trajectory', deepcopy(plan))
                    self.pr_node.get_logger().info(
                        f"[Grasp] stored grasp_trajectory to BB (points={len(plan.points)})"
                    )
                return "success"

            self.pr_node.get_logger().info("Plan found, executing...")
            exec_success = self.xarm.execute()
            if not exec_success:
                self.pr_node.get_logger().error("Execution failed.")
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
            return "success"
        else:
            self.pr_node.get_logger().warn("No valid plan found, retrying...")
            return "loop"