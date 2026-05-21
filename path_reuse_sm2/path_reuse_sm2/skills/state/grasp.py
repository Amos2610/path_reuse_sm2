#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
import rclpy
from yasmin.state import State
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from std_srvs.srv import SetBool
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper, XArmRobotUtils
from path_reuse_sm2.core.path_registry import PathRegistry


class Grasp(XArmUtilsWrapper, State):
    _approval_service = None
    _approval_event = threading.Event()
    _approval_result: Optional[bool] = None
    _cached_plan = None
    _cached_start_joint_values = None
    _cached_goal_joint_values = None

    def __init__(self, node, **kwargs):
        State.__init__(self, outcomes=["success", "loop", "except"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.kwargs = kwargs
        self.pr_client = PathSeedClient()
        self.phase = kwargs.get("phase", "")
        self.workpiece = kwargs.get("workpiece", "")
        self.path_seed_path = kwargs.get("path_seed_path", "")
        self.obj_joints = kwargs.get("joints", [])
        self.start_joints = kwargs.get("start_joints") or kwargs.get("move_joints") or []
        self.grasp_pose = kwargs.get("grasp_pose") or kwargs.get("pose")
        self.display_trajectory_pub = self.pr_node.create_publisher(
            DisplayTrajectory,
            "/display_planned_path",
            10,
        )
        self._ensure_approval_service()
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

    def _ensure_approval_service(self) -> None:
        if Grasp._approval_service is not None:
            return

        def _approval_callback(request, response):
            Grasp._approval_result = bool(request.data)
            Grasp._approval_event.set()
            response.success = True
            response.message = "grasp execution approved" if request.data else "grasp execution rejected"
            self.pr_node.get_logger().info(f"[Grasp] approval received: {response.message}")
            return response

        Grasp._approval_service = self.pr_node.create_service(
            SetBool,
            "/prsm/grasp/approve_execution",
            _approval_callback,
        )
        self.pr_node.get_logger().info(
            "[Grasp] approval service ready: /prsm/grasp/approve_execution"
        )

    def _bb_get(self, blackboard: Any, key: str, default=None):
        if blackboard is None:
            return default
        try:
            value = blackboard.get(key)
            return value if value is not None else default
        except Exception:
            try:
                value = getattr(blackboard, key)
                return value if value is not None else default
            except Exception:
                return default

    def _bb_set(self, blackboard: Any, key: str, value: Any) -> None:
        if blackboard is None:
            return
        try:
            blackboard[key] = value
        except Exception:
            try:
                setattr(blackboard, key, value)
            except Exception:
                pass

    def _param_value(self, name: str, default=None):
        try:
            return self.pr_node.get_parameter(name).value
        except Exception:
            return default

    def _as_bool(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _store_plan(
        self,
        blackboard: Any,
        plan,
        start_joint_values: List[float],
        goal_joint_values: List[float],
    ) -> None:
        Grasp._cached_plan = deepcopy(plan)
        Grasp._cached_start_joint_values = list(start_joint_values)
        Grasp._cached_goal_joint_values = list(goal_joint_values)
        self._bb_set(blackboard, "grasp_trajectory", deepcopy(plan))
        self._bb_set(blackboard, "grasp_start_joints", list(start_joint_values))
        self._bb_set(blackboard, "grasp_goal_joints", list(goal_joint_values))
        self._bb_set(blackboard, "grasp_status", "planned")
        self.pr_node.get_logger().info(
            f"[Grasp] stored grasp_trajectory to BB/cache (points={len(plan.points)})"
        )

    def _publish_display_trajectory(self, plan) -> None:
        msg = DisplayTrajectory()
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = plan
        msg.trajectory.append(robot_trajectory)
        self.display_trajectory_pub.publish(msg)
        self.pr_node.get_logger().info("[Grasp] published planned trajectory for RViz.")

    def _verify_plan_start(self, plan, start_joint_values: List[float], tolerance: float = 1.0e-3) -> None:
        if not getattr(plan, "points", None):
            self.pr_node.get_logger().warn("[Grasp] planned trajectory has no points; cannot verify start joints.")
            return

        first_positions = list(plan.points[0].positions)[:len(start_joint_values)]
        if len(first_positions) < len(start_joint_values):
            self.pr_node.get_logger().warn(
                "[Grasp] planned trajectory first point is shorter than requested start joints."
            )
            return

        max_delta = max(
            abs(float(actual) - float(expected))
            for actual, expected in zip(first_positions, start_joint_values)
        )
        if max_delta <= tolerance:
            self.pr_node.get_logger().info(
                f"[Grasp] planned trajectory starts from requested start_joints (max_delta={max_delta:.6f})."
            )
        else:
            self.pr_node.get_logger().warn(
                "[Grasp] planned trajectory first point differs from requested start_joints "
                f"(max_delta={max_delta:.6f}). MoveIt may still be planning from the current robot state."
            )

    def _set_monitoring_status(self, status: str) -> None:
        try:
            from rclpy.parameter import Parameter
            self.pr_node.set_parameters([
                Parameter("prsm_status", Parameter.Type.STRING, status),
            ])
        except Exception:
            pass

    def _wait_for_execution_approval(self, plan) -> bool:
        if not self._as_bool(self._param_value("grasp_require_approval", True), True):
            self.pr_node.get_logger().info("[Grasp] approval disabled. Executing immediately.")
            return True

        timeout_sec = float(self._param_value("grasp_approval_timeout_sec", 0.0) or 0.0)
        start_time = time.monotonic()
        next_publish_time = 0.0

        Grasp._approval_result = None
        Grasp._approval_event.clear()
        self._set_monitoring_status("planned_waiting_approval")
        self.pr_node.get_logger().info(
            "[Grasp] planned. Waiting for YES on /prsm/grasp/approve_execution..."
        )

        while rclpy.ok():
            now = time.monotonic()
            if now >= next_publish_time:
                self._publish_display_trajectory(plan)
                next_publish_time = now + 1.0

            if Grasp._approval_event.wait(timeout=0.1):
                approved = bool(Grasp._approval_result)
                if approved:
                    self.pr_node.get_logger().info("[Grasp] user approved execution.")
                else:
                    self.pr_node.get_logger().warn("[Grasp] user rejected execution.")
                return approved

            if timeout_sec > 0.0 and (now - start_time) >= timeout_sec:
                self.pr_node.get_logger().error(
                    f"[Grasp] approval timeout after {timeout_sec:.1f} sec."
                )
                return False

        self.pr_node.get_logger().error("[Grasp] rclpy shutdown while waiting for approval.")
        return False

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
        raw_obj_joints = self._bb_get(blackboard, "obj_joints", None)
        obj_joints = self.robot_utils.resolve_joints(
            raw_obj_joints,
            self.obj_joints,
            self.grasp_pose,
        )

        if not obj_joints:
            self.pr_node.get_logger().error("Blackboard missing valid 'obj_joints' and IK fallback failed.")
            return None

        self._bb_set(blackboard, "obj_joints", obj_joints)

        move_joints = self.robot_utils.resolve_joints(
            self._bb_get(blackboard, "move_joints", None),
            self._bb_get(blackboard, "start_joints", None),
            self.start_joints,
        )

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
        else:
            self.pr_node.get_logger().info(
                "Using specified start joints from blackboard/skill args."
            )

        self._bb_set(blackboard, "move_joints", move_joints)
        self._bb_set(blackboard, "start_joints", move_joints)

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
        detected_obj_joints = self._bb_get(blackboard, "obj_joints", None)

        if self.robot_utils.is_joint_list(detected_obj_joints):
            self.obj_joints = self.robot_utils.normalize_joint_list(detected_obj_joints)
            self.pr_node.get_logger().info(f"Using dynamically detected obj_joints: {self.obj_joints}")
        elif detected_obj_joints is not None:
            self.pr_node.get_logger().info("Detected obj_joints is pose-like. Defer IK to joint resolution.")
        elif self.robot_utils.is_joint_list(self.obj_joints):
            self.obj_joints = self.robot_utils.normalize_joint_list(self.obj_joints)
            self._bb_set(blackboard, "obj_joints", self.obj_joints)
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
            self.pr_node.get_logger().info("Plan found.")
            self._store_plan(blackboard, plan, start_joint_values, goal_joint_values)
            self._publish_display_trajectory(plan)
            self._verify_plan_start(plan, start_joint_values)

            if not self._wait_for_execution_approval(plan):
                self._bb_set(blackboard, "grasp_status", "execution_rejected")
                return "except"

            self.pr_node.get_logger().info("Executing approved grasp trajectory...")
            exec_success = self.xarm.execute()
            if not exec_success:
                self.pr_node.get_logger().error("Execution failed.")
                return "except"
            self.pr_node.get_logger().info("Execution succeeded.")
            self._bb_set(blackboard, "grasp_status", "executed")
            self._set_monitoring_status("running")
            self.xarm.gripper_close()
            return "success"
        else:
            self.pr_node.get_logger().warn("No valid plan found, retrying...")
            return "loop"
