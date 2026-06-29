#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
from typing import Any, Dict
from yasmin.state import State
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper


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
                self.pr_node.get_logger().info(f"[Move] Executing pre-planned trajectory for {skill_key}.")
                exec_success = self.xarm.execute_with_plan(pre_plan)
                if exec_success:
                    if blackboard is not None:
                        setattr(blackboard, "move_joints", list(pre_plan.points[-1].positions) if pre_plan.points else [])
                        return "success"
                    else:
                        self.pr_node.get_logger().error(f"[Move] Failed to set move_joints in blackboard for {skill_key}.")
                        return "except"
                else:
                    self.pr_node.get_logger().error(f"[Move] Execution of pre-planned trajectory failed for {skill_key}.")

                self.pr_node.get_logger().warn(
                    "[Move] Pre-planned execution failed (start-state mismatch?). Falling back to re-plan."
                )

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
                if blackboard is not None:
                    setattr(blackboard, "move_joints", move_joints)
                return "success"
            else:
                # デバッグ情報の追加
                try:
                    current = self.xarm.get_current_joint_values()
                except Exception as e:
                    current = None
                    self.pr_node.get_logger().warn(f"[Move] get_current_joint_values failed: {e}")

                self.pr_node.get_logger().info(f"[Move] current joints before execute: {current}")
                self.pr_node.get_logger().info(f"[Move] target joints: {move_joints}")
                jt = getattr(plan, "joint_trajectory", plan)
                self.pr_node.get_logger().info(
                    f"[Move] plan points={len(jt.points) if hasattr(jt, 'points') else 'unknown'}"
                )
                if hasattr(jt, "points") and jt.points:
                    self.pr_node.get_logger().info(f"[Move] first point={list(jt.points[0].positions)}")
                    self.pr_node.get_logger().info(f"[Move] last point={list(jt.points[-1].positions)}")


                try:
                    exec_success = self.xarm.execute()
                except Exception as e:
                    self.pr_node.get_logger().error(f"[Move] Execution failed: {e}")
                    return "loop"
                if exec_success:
                    self.pr_node.get_logger().info("Execution succeeded.")
                    if blackboard is not None:
                        setattr(blackboard, "move_joints", move_joints)
                    return "success"
                else:
                    self.pr_node.get_logger().error("Execution failed.")
                    return "loop"
        else:
            self.pr_node.get_logger().error("Planning failed.")
            return "loop"