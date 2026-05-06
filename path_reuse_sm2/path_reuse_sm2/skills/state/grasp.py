#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from yasmin.state import State
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper, XArmRobotUtils
from path_reuse_sm2.core.path_registry import PathRegistry


class Grasp(XArmUtilsWrapper, State):
    def __init__(self, node, **kwargs):
        State.__init__(self, outcomes=["success", "loop", "except"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.pr_client = PathSeedClient()
        self.phase = kwargs.get("phase", "")
        self.workpiece = kwargs.get("workpiece", "")
        self.path_seed_path = kwargs.get("path_seed_path", "")
        self.obj_joints = kwargs.get("joints", [])
        self.grasp_pose = kwargs.get("grasp_pose") or kwargs.get("pose")
        self.robot_utils = XArmRobotUtils(
            self.pr_node,
            base_frame=kwargs.get("base_frame", "link_base"),
            ee_link=kwargs.get("ee_link", "link_eef"),
            move_group=kwargs.get("move_group", "xarm6"),
            ik_service=kwargs.get("ik_service", "/compute_ik"),
            xarm=self.xarm,
        )

        # Path Registry
        self.source_id = kwargs.get("source_location", "HOME")
        # Grasp は (workpiece の場所, workpiece_id) の組み合わせで管理
        self.target_id = kwargs.get("workpiece") or kwargs.get("target_location") or ""
        self.skill_name = kwargs.get("skill_name", "SkillGraspObj")
        
        ws_root = self._get_workspace_root()
        registry_path = self.pr_node.get_parameter("pathseed_registry_path").value
        if not registry_path.startswith('/'):
            registry_path = os.path.join(ws_root, registry_path)
            
        self.registry = PathRegistry(registry_path)

        # variables
        self.try_count: int = 0
        self.pipeline: str = "stomp"  # "stomp" or "ompl"
        self.params: Dict[str, Any] = {}  # ROS1の~Params代替。必要ならbb/ファイルから供給

        # STOMP/OMPL共通のデフォルト（必要に応じて上書き）
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
            decoded_path = self.pr_client.send_decode_path_seed(file_path, start_joint_values, goal_joint_values)
            if decoded_path is None:
                self.pr_node.get_logger().error("[Grasp] decode failed: None returned.")
                return False
            self.pr_node.get_logger().info("[Grasp] Setting decoded pathseed...")
            self.pr_client.send_set_path_seed(decoded_path)

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

    def execute(self, blackboard=None):
        self.pr_node.get_logger().info("------------------------------------------------")
        self.pr_node.get_logger().info(f"Grasp state executed.")
        self.pr_node.get_logger().info("------------------------------------------------")
        self._current_blackboard = blackboard
        # 認識結果（FindObjなど）があればそれを優先、なければ RAG/KB からの値を使う
        try:
            detected_obj_joints = blackboard.get("obj_joints")
        except Exception:
            try:
                detected_obj_joints = getattr(blackboard, "obj_joints")
            except Exception:
                detected_obj_joints = None

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
        # env = blackboard.get("env", {})
        # pipeline = blackboard.get("pipeline", "stomp")

        joint_values = self.set_start_and_goal_joint_values(blackboard)
        if joint_values is None:
            self.pr_node.get_logger().error("Failed to set start/goal joint values.")
            return "except"

        start_joint_values, goal_joint_values = joint_values
        if not goal_joint_values:
            self.pr_node.get_logger().error("Failed to set joint value target.")
            return "except"
        
        # use_pathseed = True  # TODO: blackboard等で切り替え可能に
        use_pathseed = self.pr_node.get_parameter("use_pathseed").value
        self.pr_node.get_logger().info(f"use_pathseed: {use_pathseed}")
        if use_pathseed:
            self.xarm.set_planning_pipeline("stomp")
            # PathSeedからSTOMP用軌道をセット
            self.xarm.set_move_group_parameter("stomp.use_custom_trajectory", True)
            self.pr_node.get_logger().info(f"Grasp phase: {self.phase}")
            # 1. Registry から検索
            reg_path = self.registry.get_path_seed(self.source_id, self.target_id, self.skill_name)
            
            if reg_path:
                self.pr_node.get_logger().info(f"[Grasp] Found entry in registry: {reg_path}")
                # "ex1_..." 形式なら prefix を補完
                if not reg_path.startswith("src/") and not reg_path.startswith("/"):
                    pathseed_file = "src/path_reuse_method/pathseeds/Library/" + reg_path
                else:
                    pathseed_file = reg_path
            # 2. RAG からの直接指定
            elif self.path_seed_path:
                self.pr_node.get_logger().info(f"[Grasp] Using path_seed_path from RAG: {self.path_seed_path}")
                pathseed_file = self.path_seed_path
            # 3. ROS パラメータ（Phase に応じる）
            elif self.phase == "Initial_Phase":
                pathseed_file = self.pr_node.get_parameter("pathseed_grasp").value
            elif self.phase == "Imprementation_Phase":
                default_pathseed_file = self.pr_node.get_parameter("pathseed_grasp").value
                # updateしたパスシードはex1_pick_and_place/updated/pathseed_pick.txtに保存される想定
                pathseed_file = "/".join(default_pathseed_file.split("/")[:-3]) + "/updated/pathseed_pick.txt"
            else:
                self.pr_node.get_logger().error(f"Unknown phase: {self.phase}")
                return "except"
                
            # 絶対パスに変換
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
            self.xarm.set_planning_pipeline("ompl")
        
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
                setattr(self._current_blackboard, 'grasp_trajectory', deepcopy(plan))
                self.pr_node.get_logger().info(f"[Grasp] stored grasp_trajectory to BB (points={len(plan.points)})")
            self.xarm.gripper_close()
            return "success"
        else:
            self.pr_node.get_logger().warn("No valid plan found, retrying...")
            return "loop"