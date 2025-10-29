#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any
from yasmin.state import State
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper


class FindObj(State):
    def __init__(self, node):
        super().__init__(outcomes=["success", "except", "loop"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.pr_client = PathSeedClient()

    def execute(self, blackboard: Any = None) -> str:
        self.pr_node.get_logger().info("FindObj state executed.")
        obj_pose = [0.916, 0.724, -1.70014, 0.001, 0.977, -0.67]  # TODO: 仮の値
        # container_pose = [130, 47, -107, 0, 60, 34] # degrees, TODO: 仮の値
        # container_pose = [val * 3.14159265 / 180.0 for val in container_pose]  # radian
        # self.pr_node.get_logger().info(f"Container pose (rad): {container_pose}")
        container_pose = [2.268928025, 0.8203047475, -1.8675022975, 0.0, 1.0471975500000001, 0.593411945]  # TODO: 仮の値

        # DEBUG
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

        # TODO: ここに認識処理を実装していく

        # Blackboardに認識結果をセット
        self.pr_node.get_logger().info("[FindObj] set obj_pose")
        setattr(blackboard, "obj_pose", obj_pose) #TODO 一旦C空間のままセット
        setattr(blackboard, "container_pose", container_pose) #TODO 一旦C空間のままセット
        return "success"
