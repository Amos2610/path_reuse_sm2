#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from yasmin.state import State
from rclpy.action import ActionServer
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.action import ExecuteTrajectory
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper


class Put(XArmUtilsWrapper, State):
    def __init__(self, node, approach_margin=0.01):
        State.__init__(self, outcomes=["success", "loop", "except"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.pr_client = PathSeedClient()
        # ActionServerの初期化
        self._action_server = ActionServer(
            self.pr_node,
            ExecuteTrajectory,
            '/prsm/put/execute_trajectory',
            self.execute_trajectory_callback
        )
        self._last_executed_traj_msg: Optional[JointTrajectory] = None
        self.approach_margin = approach_margin

        # ROS1互換フィールド名
        self.try_count: int = 0
        self.pipeline: str = "stomp"  # "stomp" or "ompl"
        self.params: Dict[str, Any] = {}  # ROS1の~Params代替。必要ならbb/ファイルから供給

        # STOMP/OMPL共通のデフォルト（必要に応じて上書き）
        self.max_retries_default = 2
        self.max_velocity_scale_default = 0.3
        self.max_accel_scale_default = 0.3
        self.planning_time_default = 2.0

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
            self.pr_node.get_logger().info("[Put] Decoding pathseed via PathSeedClient...")
            decoded_path = self.pr_client.send_decode_path_seed(file_path, start_joint_values, goal_joint_values)
            if decoded_path is None:
                self.pr_node.get_logger().error("[Put] decode failed: None returned.")
                return False
            self.pr_node.get_logger().info("[Put] Setting decoded pathseed...")
            self.pr_client.send_set_path_seed(decoded_path)

            # デバッグ用に残したい場合はBBへ保存する運用に
            # bb.pathseed_decoded = decoded_path

            return True
        except Exception as e:
            self.pr_node.get_logger().error(f"[Put] Exception in generate_stomp_path_from_pathseed: {e}")
            return False

    def set_start_and_goal_joint_values(self, blackboard: Any) -> Optional[Tuple[List[float], List[float]]]:
        obj_pose = getattr(blackboard, "obj_pose", None)
        if obj_pose is None:
            self.pr_node.get_logger().error("Blackboard missing 'obj_pose'.")
            return None
        
        container_pose = getattr(blackboard, "container_pose", None)
        if container_pose is None:
            self.pr_node.get_logger().error("Blackboard missing 'container_pose'.")
            return None
        
        #TODO: ここでcontainer_poseからTF変換，逆運動学を使ってstart_joint_valuesを計算する
        # 一旦container_poseのまま使う
        goal_joint_values = container_pose
        self.pr_node.get_logger().info(f"Put goal joint values: {goal_joint_values}")
        
        #TODO: ここでobj_poseからTF変換，逆運動学を使ってgoal_joint_valuesを計算する
        # 一旦obj_poseのまま使う
        start_joint_values = obj_pose
        self.pr_node.get_logger().info(f"Put start joint values: {start_joint_values}")
        return start_joint_values, goal_joint_values

    def execute(self, blackboard=None):
        self.pr_node.get_logger().info("------------------------------------------------")
        self.pr_node.get_logger().info(f"Put state executed. approach_margin={self.approach_margin}")
        self.pr_node.get_logger().info("------------------------------------------------")
        self._current_blackboard = blackboard
        self.phase = self.pr_node.get_parameter("put_phase").value
        # env = blackboard.get("env", {})
        # pipeline = blackboard.get("pipeline", "stomp")
        # self.xarm.set_planning_pipeline("ompl")
        self.xarm.set_planning_pipeline("stomp")
        start_joint_values, goal_joint_values = self.set_start_and_goal_joint_values(blackboard)
        if not goal_joint_values:
            self.pr_node.get_logger().error("Failed to set joint value target.")
            return "except"
        
        # use_pathseed = True  # TODO: blackboard等で切り替え可能に
        use_pathseed = self.pr_node.get_parameter("use_pathseed").value
        if use_pathseed:
            # PathSeedからSTOMP用軌道をセット
            self.xarm.set_move_group_parameter("stomp.use_custom_trajectory", True)
            self.pr_node.get_logger().info(f"Put phase: {self.phase}")
            if self.phase == "Initial_Phase":
                pathseed_file = self.pr_node.get_parameter("pathseed_put").value
            elif self.phase == "Imprementation_Phase":
                default_pathseed_file = self.pr_node.get_parameter("pathseed_put").value
                # updateしたパスシードはex1_pick_and_place/updated/pathseed_pick.txtに保存される想定
                pathseed_file = "/".join(default_pathseed_file.split("/")[:-3]) + "/updated/pathseed_place.txt"
            else:
                self.pr_node.get_logger().error(f"Unknown phase: {self.phase}")
                return "except"
            self.pr_node.get_logger().info(f"[Put] Using pathseed file: {pathseed_file}")
            success_generated = self.generate_stomp_path_from_pathseed(
                file_path=pathseed_file,
                start_joint_values=start_joint_values,
                goal_joint_values=goal_joint_values
            )
            if not success_generated:
                self.pr_node.get_logger().error("Failed to generate STOMP path from PathSeed.")
                return "except"

        ##############################
        ### Planning and Execution ###
        ##############################
        self.xarm.set_joint_value_target(goal_joint_values)
        success, plan, _, _ = self.xarm.plan()
        if success:
            self.pr_node.get_logger().info("Plan found, executing...")
            exec_success = self.xarm.execute()
            if not exec_success:
                self.pr_node.get_logger().error("Execution failed.")
                return "except"
            self.pr_node.get_logger().info("Execution succeeded.")
            # Blackboardに軌道を保存
            if self._current_blackboard is not None:
                setattr(self._current_blackboard, 'put_trajectory', deepcopy(plan))
                self.pr_node.get_logger().info(f"[Put] stored put_trajectory to BB (points={len(plan.points)})")
            return "success"
        else:
            self.pr_node.get_logger().warn("No valid plan found, retrying...")
            return "loop"