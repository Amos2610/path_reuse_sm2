#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict
from path_reuse_sm2.core.state import SkillState


class Move(SkillState):
    def __init__(self, node):
        super().__init__(node, name="Move", outcomes=["success"])

    def run(self, bb: Dict[str, Any], **_) -> str:
        self.node.get_logger().info("Move state executed.")
        # parameter 取得
        self.declare_parameter("move_target_joints", None)
        self.declare_parameter("move_target_pose", None)
        target_joints = self.node.get_parameter("move_target_joints").get_parameter_value().string_value
        target_pose = self.node.get_parameter("move_target_pose").get_parameter_value().string_value
        self.node.get_logger().info(f"Moving to pose: {target_pose}, joints: {target_joints}")

        # target_joints（C-space）があればそちらを優先
        if target_joints is not None and target_joints != "":
            self.node.get_logger().info(f"Moving to target joints: {target_joints}")
            # ここに関節位置への移動コードを追加
        elif target_pose is not None and target_pose != "":
            self.node.get_logger().info(f"Moving to target pose: {target_pose}")
            # ここにエンドエフェクタ位置への移動コードを追加
        else:
            self.node.get_logger().warn("No target joints or pose specified. Skipping move.")
        return "success"