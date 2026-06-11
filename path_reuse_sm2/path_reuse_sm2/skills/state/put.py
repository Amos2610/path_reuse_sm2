#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from yasmin.state import State
from rclpy.action import ActionServer
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper, XArmRobotUtils
from path_reuse_sm2.core.path_registry import PathRegistry
import os


class Put(XArmUtilsWrapper, State):
    def __init__(self, node, **kwargs):
        State.__init__(self, outcomes=["success", "loop", "except"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.kwargs = kwargs
        self.pr_client = None
        self._current_blackboard = None

        # ActionServerの初期化
        self._action_server = ActionServer(
            self.pr_node,
            ExecuteTrajectory,
            '/prsm/put/execute_trajectory',
            self.execute_trajectory_callback
        )
        self._last_executed_traj_msg: Optional[JointTrajectory] = None
        self.target_location = kwargs.get("target_location", "")
        self.joints = kwargs.get("joints", [])
        self.pose = kwargs.get("pose", [])
        self.path_seed_path = kwargs.get("path_seed_path", "")
        self.step_index = kwargs.get("step_index", 0)

        # Path Registry
        self.source_id = kwargs.get("source_location", "UNKNOWN")
        self.target_id = kwargs.get("target_location") or "CONTAINER"
        self.skill_name = kwargs.get("skill_name", "SkillPutObj")
        
        ws_root = self._get_workspace_root()
        registry_path = self.pr_node.get_parameter("pathseed_registry_path").value
        if not registry_path.startswith('/'):
            registry_path = os.path.join(ws_root, registry_path)
            
        self.registry = PathRegistry(registry_path)

        self.robot_utils = XArmRobotUtils(
            self.pr_node,
            base_frame=kwargs.get("base_frame", "link_base"),
            ee_link=kwargs.get("ee_link", "link_tcp"),
            move_group=kwargs.get("move_group", "xarm6"),
            ik_service=kwargs.get("ik_service", "/compute_ik"),
            xarm=self.xarm,
        )

        # ROS1互換フィールド名
        self.try_count: int = 0
        self.pipeline: str = "stomp"  # "stomp" or "ompl"
        self.params: Dict[str, Any] = {}  # ROS1の~Params代替．必要ならbb/ファイルから供給

        # STOMP/OMPL共通のデフォルト，必要に応じて上書き
        self.max_retries_default = 2
        self.max_velocity_scale_default = 0.3
        self.max_accel_scale_default = 0.3
        self.planning_time_default = 2.0

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

    def _publish_display_trajectory(self, plan_jt) -> None:
        """simulate_only 時に軌道をループパブリッシャーに渡す．"""
        if not hasattr(self.pr_node, '_viz_traj_loop'):
            return
        from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = plan_jt
        disp = DisplayTrajectory()
        disp.model_id = "xarm6"
        disp.trajectory.append(robot_traj)
        interval = 5.0
        if plan_jt.points:
            last = plan_jt.points[-1].time_from_start
            duration = last.sec + last.nanosec * 1e-9
            interval = max(duration + 1.0, 3.0)
        self.pr_node._viz_traj_loop.start(disp, interval)
        self.pr_node.get_logger().info(
            f"[Put] Trajectory loop started (interval={interval:.1f}s, {len(plan_jt.points)} points)."
        )

    async def execute_trajectory_callback(self, goal_handle):
        self.pr_node.get_logger().info("Received ExecuteTrajectory goal.")
        try:
            trajectory: JointTrajectory = goal_handle.request.trajectory.joint_trajectory
            self._last_executed_traj_msg = trajectory
            # Blackboardに反映
            if self._current_blackboard is not None:
                setattr(self._current_blackboard, 'put_trajectory', trajectory)
            goal_handle.succeed()
            return ExecuteTrajectory.Result()
        except Exception as e:
            self.pr_node.get_logger().error(f"Error executing trajectory: {e}")
            goal_handle.abort()
            return ExecuteTrajectory.Result()

    # ====== set_stomp_params ======
    # 目的: 与えられたphase(dict)をMoveItへ適用，ROS2ではrosparamでなくAPI直適用
    def set_stomp_params(self, phase: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set STOMP parameters into MoveIt via XArmUtils.set_move_group_parameter
        Args:
            phase: フェーズごとのSTOMPパラメータ辞書，ROS1互換
        Returns:
            適用したパラメータ，検証ログ用に返す
        """
        # 代表的なキー例．存在するものだけ適用，ROS1 -> ROS2移行の最小公倍数
        mapping = {
            # 速度・加速度・計画時間，STOMP/OMPL共通でも使える
            "planning_time": "planning_time",
            "max_velocity_scaling_factor": "max_velocity_scaling_factor",
            "max_acceleration_scaling_factor": "max_acceleration_scaling_factor",

            # 以下はSTOMP拡張，MoveIt側のパラメータ名に合わせて適宜調整
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
                # 代表的な型に合わせて設定，bool/int/float/str対応
                if isinstance(v, bool):
                    self.xarm.set_move_group_parameter(k_dst, bool(v))
                elif isinstance(v, int):
                    self.xarm.set_move_group_parameter(k_dst, int(v))
                elif isinstance(v, float):
                    self.xarm.set_move_group_parameter(k_dst, float(v))
                elif isinstance(v, str):
                    self.xarm.set_move_group_parameter(k_dst, str(v))
        return phase

    def _ensure_path_seed_client(self) -> PathSeedClient:
        if self.pr_client is None:
            self.pr_node.get_logger().info("[Put] Creating PathSeedClient.")
            self.pr_client = PathSeedClient()
        return self.pr_client

    def generate_stomp_path_from_pathseed(
        self,
        file_path: str,
        start_joint_values: List[float],
        goal_joint_values: List[float],
    ) -> bool:
        """
        PathSeed ファイルを decode → set する，ROS2版
        ROS1のDecoder+param set の代わりに PathSeedClient を使用
        """
        try:
            pr_client = self._ensure_path_seed_client()
            self.pr_node.get_logger().info("[Put] Decoding pathseed via PathSeedClient...")
            decoded_path = pr_client.send_decode_path_seed(file_path, start_joint_values, goal_joint_values)
            if decoded_path is None:
                self.pr_node.get_logger().error("[Put] decode failed: None returned.")
                return False
            self.pr_node.get_logger().info("[Put] Setting decoded pathseed...")
            pr_client.send_set_path_seed(decoded_path)

            # デバッグ用に残したい場合はBBへ保存する運用に
            # bb.pathseed_decoded = decoded_path

            return True
        except Exception as e:
            self.pr_node.get_logger().error(f"[Put] Exception in generate_stomp_path_from_pathseed: {e}")
            return False

    def set_start_and_goal_joint_values(self, blackboard: Any) -> Optional[Tuple[List[float], List[float]]]:
        # Start = current arm position，the robot is holding the object here
        start_joint_values = self.robot_utils.get_current_joint_values()
        if start_joint_values is None:
            # Fall back to obj_joints on blackboard if current joints unavailable，e.g. fake mode
            obj_joints = blackboard["obj_joints"] if "obj_joints" in blackboard else None
            if obj_joints:
                self.pr_node.get_logger().warn("[Put] Current joint values unavailable. Using obj_joints from BB as start.")
                start_joint_values = obj_joints
            else:
                self.pr_node.get_logger().error("[Put] Cannot determine start joint values.")
                return None
        self.pr_node.get_logger().info(f"Put start joint values: {start_joint_values}")

        # Goal = container/target position
        container_joints = []
        if self.joints:
            self.pr_node.get_logger().info(f"Moving to target joints from RAG: {self.joints}")
            container_joints = self.joints
        elif self.pose:
            self.pr_node.get_logger().info(f"Moving to target pose from RAG: {self.pose}")
            container_joints = blackboard["container_joints"] if "container_joints" in blackboard else None
            # TODO: IK計算して関節角度に変換する処理を実装する
            # container_joints = self.convert_pose_to_joints(self.pose)
        else:
            container_joints = [2.268928025, 0.8203047475, -1.8675022975, 0.0, 1.0471975500000001, 0.593411945]

        if container_joints is None:
            self.pr_node.get_logger().error("Blackboard missing 'container_joints'.")
            return None

        goal_joint_values = container_joints
        self.pr_node.get_logger().info(f"Put goal joint values: {goal_joint_values}")
        return start_joint_values, goal_joint_values

    def execute(self, blackboard=None):
        self.pr_node.get_logger().info("------------------------------------------------")
        self.pr_node.get_logger().info(f"Put state executed.")
        self.pr_node.get_logger().info("------------------------------------------------")
        self._current_blackboard = blackboard

        def _bb_get(key, default=None):
            if blackboard is None:
                return default
            try:
                value = blackboard.get(key)
                return value if value is not None else default
            except Exception:
                pass
            try:
                return blackboard[key]
            except Exception:
                pass
            try:
                value = getattr(blackboard, key)
                return value if value is not None else default
            except Exception:
                return default

        def _bb_set(key, value):
            if blackboard is None:
                return
            try:
                blackboard[key] = value
                return
            except Exception:
                pass
            try:
                setattr(blackboard, key, value)
            except Exception:
                pass

        self.phase = self.pr_node.get_parameter("put_phase").value

        ##############################
        ### Fast path: pre-planned ###
        ##############################
        simulate_only = self.pr_node.get_parameter("prsm_simulate_only").value
        skill_key = f"{self.skill_name}_{self.step_index}"

        if not simulate_only:
            pre_plans = {}
            try:
                # Blackboard の複数のアクセス方法を試す
                pre_plans = blackboard.get("pre_planned_trajectories")
                if pre_plans is None:
                    pre_plans = {}
            except KeyError:
                # キーが存在しない場合
                self.pr_node.get_logger().warn(
                    "[Put] 'pre_planned_trajectories' not in blackboard. This is expected in first execution."
                )
                pre_plans = {}
            except Exception as e:
                self.pr_node.get_logger().warn(f"[Put] Blackboard access failed: {e}. Continuing with empty dict.")
                pre_plans = getattr(blackboard, "pre_planned_trajectories", {}) or {}

            pre_plan = pre_plans.get(skill_key)
            if pre_plan is not None:
                self.pr_node.get_logger().info(f"[Put] Executing pre-planned trajectory for {skill_key}.")
                exec_success = self.xarm.execute_with_plan(pre_plan)
                if not exec_success:
                    self.pr_node.get_logger().error("[Put] Pre-planned execution failed.")
                    return "except"
                if self._current_blackboard is not None:
                    setattr(self._current_blackboard, 'put_trajectory', deepcopy(pre_plan))
                self.xarm.gripper_open()
                return "success"

        ######################################
        ### Normal path: resolve → plan → execute/simulate ###
        ######################################
        # Blackboard に obj_joints がない場合は，自身の kwargs，RAG由来，から補完を試みる
        if blackboard is not None and _bb_get("obj_joints", None) is None:
            joints_from_rag = self.kwargs.get("joints")
            if joints_from_rag:
                self.pr_node.get_logger().info(f"[Put] Syncing obj_joints to BB from RAG args: {joints_from_rag}")
                _bb_set("obj_joints", joints_from_rag)

        joint_values = self.set_start_and_goal_joint_values(blackboard)
        if joint_values is None:
            self.pr_node.get_logger().error("Failed to set start/goal joint values.")
            return "except"

        start_joint_values, goal_joint_values = joint_values
        if not goal_joint_values:
            self.pr_node.get_logger().error("Failed to set joint value target.")
            return "except"

        use_pathseed = self.pr_node.get_parameter("use_pathseed").value
        if use_pathseed:
            pr_client = self._ensure_path_seed_client()
            self.xarm.set_planning_pipeline("stomp")
            self.xarm.set_move_group_parameter("stomp.use_custom_trajectory", True)
            self.pr_node.get_logger().info(f"Put phase: {self.phase}")

            if self.path_seed_path:
                pathseed_file = self.path_seed_path
            else:
                pathseed_file = pr_client.select_best_path_seed(
                    environment_id="desk_scene_v1",
                    skill_name="place",
                    start_joints=start_joint_values,
                    goal_joints=goal_joint_values,
                )
                if not pathseed_file:
                    if self.phase == "Initial_Phase":
                        pathseed_file = self.pr_node.get_parameter("pathseed_put").value
                    elif self.phase in ("Implement_Phase", "Imprementation_Phase"):
                        default_pathseed_file = self.pr_node.get_parameter("pathseed_put").value
                        pathseed_file = "/".join(default_pathseed_file.split("/")[:-3]) + "/updated/pathseed_place.txt"
                    else:
                        self.pr_node.get_logger().error(f"Unknown phase: {self.phase}")
                        return "except"

            if not pathseed_file.startswith('/'):
                pathseed_file = os.path.join(self._get_workspace_root(), pathseed_file)
            self.pr_node.get_logger().info(f"[Put] Using pathseed file: {pathseed_file}")
            success_generated = self.generate_stomp_path_from_pathseed(
                file_path=pathseed_file,
                start_joint_values=start_joint_values,
                goal_joint_values=goal_joint_values
            )
            if not success_generated:
                self.pr_node.get_logger().error("Failed to generate STOMP path from PathSeed.")
                return "except"
        else:
            self.xarm.set_planning_pipeline("ompl")

        self.xarm.set_joint_value_target(goal_joint_values)
        success, plan, _, _ = self.xarm.plan()
        if success:
            if simulate_only:
                self.pr_node.get_logger().info(f"[Put] Storing pre-planned trajectory for {skill_key}.")
                self._publish_display_trajectory(plan)
                self.pr_node._pre_planned_trajectories[skill_key] = plan
                if self._current_blackboard is not None:
                    setattr(self._current_blackboard, 'put_trajectory', deepcopy(plan))
                return "success"

            self.pr_node.get_logger().info("Plan found, executing...")
            exec_success = self.xarm.execute()
            if not exec_success:
                self.pr_node.get_logger().error("Execution failed.")
                return "except"
            self.pr_node.get_logger().info("Execution succeeded.")
            if self._current_blackboard is not None:
                setattr(self._current_blackboard, 'put_trajectory', deepcopy(plan))
            self.xarm.gripper_open()
            return "success"
        else:
            self.pr_node.get_logger().warn("No valid plan found, retrying...")
            return "loop"