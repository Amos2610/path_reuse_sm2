#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict
from yasmin.state import State
from xarm_utils_py import XArmUtils , Node
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper


class Put(XArmUtilsWrapper, State):
    def __init__(self, node, approach_margin=0.01):
        State.__init__(self, outcomes=["success", "loop", "except"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.approach_margin = approach_margin

    def execute(self, blackboard=None):
        self.pr_node.get_logger().info("------------------------------------------------")
        self.pr_node.get_logger().info(f"Put state executed. approach_margin={self.approach_margin}")
        self.pr_node.get_logger().info("------------------------------------------------")

        # self.phase = blackboard.get("phase", "Initial_Phase")
        # env = blackboard.get("env", {})
        # pipeline = blackboard.get("pipeline", "stomp")
        self.xarm.set_planning_pipeline("ompl")

        # goal_joint_values = self.xarm.set_joint_value_target(env.get("goal_joint_values", []))
        goal_joint_values = [0, 0, 0, 0, 0, 0]  # degrees
        goal_joint_values = [val * 3.14159265 / 180.0 for val in goal_joint_values]  # to radians
        if not goal_joint_values:
            self.pr_node.get_logger().error("Failed to set joint value target.")
            return "except"
        self.xarm.set_joint_value_target(goal_joint_values)
        success, plan, _, _ = self.xarm.plan()
        if success:
            self.pr_node.get_logger().info("Plan found, executing...")
            exec_success = self.xarm.execute()
            if not exec_success:
                self.pr_node.get_logger().error("Execution failed.")
                return "except"
            self.pr_node.get_logger().info("Execution succeeded.")
            return "success"
        else:
            self.pr_node.get_logger().warn("No valid plan found, retrying...")
            return "loop"