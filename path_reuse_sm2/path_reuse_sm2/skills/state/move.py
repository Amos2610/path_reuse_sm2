#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    def execute(self, blackboard=None, **_) -> str:
        self.pr_node.get_logger().info("Move state executed.")
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
            self.pr_node.get_logger().info("Moved to obj_joints for debug.")
            result = self.xarm.plan()
            success = result[0] if isinstance(result, (list, tuple)) and len(result) >= 1 else False
        except Exception as e:
            self.pr_node.get_logger().error(f"[Move] Motion planning failed (is move_group running?): {e}")
            return "except"

        if success:
            self.pr_node.get_logger().info("Planning to obj_joints succeeded.")
            try:
                exec_success = self.xarm.execute()
            except Exception as e:
                self.pr_node.get_logger().error(f"[Move] Execution failed: {e}")
                return "loop"
            if exec_success:
                self.pr_node.get_logger().info("Execution to obj_joints succeeded.")
                setattr(blackboard, "move_joints", move_joints)
                return "success"
            else:
                self.pr_node.get_logger().error("Execution to obj_joints failed.")
                return "loop"
        else:
            self.pr_node.get_logger().error("Planning to obj_joints failed.")
            return "loop"