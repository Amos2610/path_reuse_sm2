#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Skill: Put Object
"""
import rclpy
from typing import Any, Dict
from yasmin.state import State
from yasmin_ros.basic_outcomes import SUCCEED
from path_reuse_sm2.core.plugin import skill
from path_reuse_sm2_interfaces.msg import TaskInfo
from path_reuse_sm2_interfaces.srv import TaskGet


@skill("SkillStart")
class SkillStart(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED])
        self.node = node

    def execute(self, blackboard: Dict[str, Any]) -> str:
        """
        Start Skill: Initialize task-related blackboard entries
        """
        self.node.get_logger().info("[SkillStart] Start")

        task: TaskInfo = blackboard.get("task", None)
        if task is None:
            self.node.get_logger().error("[SkillStart] blackboard['task'] is None")
            return SUCCEED

        self.node.get_logger().info(
            f"[SkillStart] Current Task has {len(task.skills)} skills."
        )

        # 例えば最初のスキルインデックス初期化など
        blackboard["current_skill_index"] = 0

        return SUCCEED
