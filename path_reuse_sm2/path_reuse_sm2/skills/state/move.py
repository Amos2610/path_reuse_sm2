#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any, Dict
from yasmin.state import State
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper
from path_reuse_sm2.core import plan_helpers as ph


class Move(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=["success", "except", "loop"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.target_location = kwargs.get("target_location", "")
        self.joints = kwargs.get("joints", [])
        self.pose = kwargs.get("pose", [])
        self.path_seed_path = kwargs.get("path_seed_path", "")
        self.step_index = kwargs.get("step_index", 0)
        self.skill_name = kwargs.get("skill_name", "SkillMove")

        self.try_count: int = 0
        self.max_retries_default = 2

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
            f"[Move] Trajectory loop started (interval={interval:.1f}s, {len(plan_jt.points)} points)."
        )

    def execute(self, blackboard=None, **_) -> str:
        self.pr_node.get_logger().info("Move state executed.")

        simulate_only = self.pr_node.get_parameter("prsm_simulate_only").value
        skill_key = f"{self.skill_name}_{self.step_index}"

        ##############################
        ### Fast path: pre-planned ###
        ##############################
        if not simulate_only:
            pre_plans = blackboard["pre_planned_trajectories"] if "pre_planned_trajectories" in blackboard else {}
            pre_plan = pre_plans.get(skill_key)
            if pre_plan is not None:
                # 事前計画軌道はシミュレーション時の開始姿勢から作られている。承認までの
                # 間に腕が動いていると、MoveIt の開始点検証（allowed_start_tolerance）が
                # 実行を弾いて abort になる。弾かれてから諦めるのではなく、ずれを先に
                # 検知したら事前計画を捨てて現在姿勢から計画し直す。
                stale = False
                try:
                    current = list(self.xarm.get_current_joint_values() or [])
                    start = list(pre_plan.points[0].positions) if pre_plan.points else []
                    if current and start and len(current) == len(start):
                        deviation = max(abs(a - b) for a, b in zip(current, start))
                        if deviation > 0.05:
                            stale = True
                            self.pr_node.get_logger().warn(
                                f"[Move] 事前計画軌道の開始点が現在姿勢から {deviation:.3f} rad "
                                "ずれている。事前計画を捨てて現在姿勢から再計画する"
                            )
                except Exception as e:  # noqa: BLE001 - 検証に失敗したら従来どおり再生に賭ける
                    self.pr_node.get_logger().warn(
                        f"[Move] 事前計画軌道の開始点検証に失敗: {e}（そのまま再生する）"
                    )
                if not stale:
                    self.pr_node.get_logger().info(f"[Move] Executing pre-planned trajectory for {skill_key}.")
                    exec_success = self.xarm.execute_with_plan(pre_plan)
                    if not exec_success:
                        self.pr_node.get_logger().error("[Move] Pre-planned execution failed.")
                        return "except"
                    if blackboard is not None:
                        setattr(blackboard, "move_joints", list(pre_plan.points[-1].positions) if pre_plan.points else [])
                    return "success"
                # stale: 下の通常経路（plan → execute）へ落ちる

        ######################################
        ### Normal path: plan → simulate/execute ###
        ######################################
        try:
            self.xarm.set_planning_pipeline("ompl")
        except Exception as e:
            self.pr_node.get_logger().error(f"[Move] Failed to set planning pipeline: {e}")
            return "except"

        move_joints = []
        if self.joints:
            self.pr_node.get_logger().info(f"Moving to target joints from RAG: {self.joints}")
            move_joints = self.joints
        elif self.pose:
            self.pr_node.get_logger().info(f"Moving to target pose from RAG: {self.pose}")
            # TODO: IK計算して関節角度に変換する処理を実装する
        else:
            self.pr_node.get_logger().error("No valid joints or pose provided. Please define them in semantic_kb.")
            return "except"

        try:
            ph.apply_start_state(self.xarm, self.pr_node, simulate_only, ph.sim_end_joints(blackboard), "Move")
            self.xarm.set_joint_value_target(move_joints)
            result = self.xarm.plan()
            success = result[0] if isinstance(result, (list, tuple)) and len(result) >= 1 else False
            plan = result[1] if isinstance(result, (list, tuple)) and len(result) >= 2 else None
        except Exception as e:
            self.pr_node.get_logger().error(f"[Move] Motion planning failed (is move_group running?): {e}")
            return "except"

        if success:
            self.pr_node.get_logger().info("Planning succeeded.")
            if simulate_only:
                self.pr_node.get_logger().info(f"[Move] Storing pre-planned trajectory for {skill_key}.")
                if plan is not None:
                    self.pr_node._pre_planned_trajectories[skill_key] = plan
                    self._publish_display_trajectory(plan)
                    if plan.points:
                        ph.set_sim_end_joints(blackboard, plan.points[-1].positions)
                if blackboard is not None:
                    setattr(blackboard, "move_joints", move_joints)
                self.try_count = 0
                return "success"
            else:
                try:
                    exec_success = self.xarm.execute()
                except Exception as e:
                    self.pr_node.get_logger().error(f"[Move] Execution failed: {e}")
                    return self._retry_or_abort("execution raised")
                if exec_success:
                    self.pr_node.get_logger().info("Execution succeeded.")
                    if blackboard is not None:
                        setattr(blackboard, "move_joints", move_joints)
                    self.try_count = 0
                    return "success"
                else:
                    return self._retry_or_abort("execution failed")
        else:
            return self._retry_or_abort("planning failed")

    def _retry_or_abort(self, reason: str) -> str:
        """"loop" は skill_move.py で MOVE へ戻るので、上限が無いと永久に返らない。"""
        self.try_count += 1
        if self.try_count > self.max_retries_default:
            self.pr_node.get_logger().error(
                f"[Move] {reason} after {self.try_count} attempts. Aborting skill."
            )
            self.try_count = 0
            return "except"
        self.pr_node.get_logger().warn(
            f"[Move] {reason} ({self.try_count}/{self.max_retries_default}), retrying..."
        )
        return "loop"