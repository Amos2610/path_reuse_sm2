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
        self.xarm.set_planning_pipeline("ompl")
        move_joints = []
        # jointsの値がある場合は優先して使用する
        if self.joints:
            self.pr_node.get_logger().info(f"Moving to target joints from RAG: {self.joints}")
            move_joints = self.joints
        # jointsの値は空でposeの値がある場合はposeをIK変換して使用する
        elif not self.joints and self.pose:
            self.pr_node.get_logger().info(f"Moving to target pose from RAG: {self.pose}")
            # move_joints = [2.268928025, 0.8203047475, -1.8675022975, 0.0, 1.0471975500000001, 0.593411945]
            # deg[-8, 0, -49 -2, 50, 0] deg -> rad
            # rad[-0.13962634015954636, 0.0, -0.8552113334772214, -0.03490658503988659, 0.8726646259971648, 0.0]
            move_joints = [-0.13962634015954636, 0.0, -0.8552113334772214, -0.03490658503988659, 0.8726646259971648, 0.0]
            # TODO: IK計算して関節角度に変換する処理を実装する
            # move_joints = self.convert_pose_to_joints(self.pose)
        else:
            # move_joints = [2.268928025, 0.8203047475, -1.8675022975, 0.0, 1.0471975500000001, 0.593411945]
            move_joints = [-0.13962634015954636, 0.0, -0.8552113334772214, -0.03490658503988659, 0.8726646259971648, 0.0]

        self.xarm.set_joint_value_target(move_joints)
        self.pr_node.get_logger().info("Moved to obj_joints for debug.")
        success, plan, _, _ = self.xarm.plan()
        if success:
            self.pr_node.get_logger().info("Planning to obj_joints succeeded.")
            exec_success = self.xarm.execute()
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
        return "except"