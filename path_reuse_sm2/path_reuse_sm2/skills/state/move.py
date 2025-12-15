#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict
from yasmin.state import State
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper


class Move(State):
    def __init__(self, node):
        super().__init__(outcomes=["success", "except"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node

    def execute(self, blackboard=None, **_) -> str:
        self.pr_node.get_logger().info("Move state executed.")
        # DEBUG
        container_pose = [2.268928025, 0.8203047475, -1.8675022975, 0.0, 1.0471975500000001, 0.593411945]  # TODO: 仮の値
        self.xarm.set_planning_pipeline("ompl")

        self.xarm.set_joint_value_target(container_pose)
        self.pr_node.get_logger().info("Moved to obj_pose for debug.")
        success, plan, _, _ = self.xarm.plan()
        if success:
            self.pr_node.get_logger().info("Planning to obj_pose succeeded.")
            exec_success = self.xarm.execute()
            if exec_success:
                self.pr_node.get_logger().info("Execution to obj_pose succeeded.")
            else:
                self.pr_node.get_logger().error("Execution to obj_pose failed.")
        else:
            self.pr_node.get_logger().error("Planning to obj_pose failed.")

        setattr(blackboard, "container_pose", container_pose) #TODO 一旦C空間のままセット
        return "success"