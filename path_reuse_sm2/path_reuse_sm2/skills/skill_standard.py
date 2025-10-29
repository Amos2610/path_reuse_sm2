#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Skill: Put Object
"""
from yasmin.state import State
from yasmin_ros.basic_outcomes import SUCCEED
from path_reuse_sm2.core.plugin import skill


@skill("SkillStart")
class SkillStart(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED])
        self.node = node

    def execute(self, blackboard):
        self.node.get_logger().info("[SkillStart] Start (dummy)")
        return SUCCEED
