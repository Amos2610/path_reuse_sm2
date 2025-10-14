#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Skill: Put Object
"""
from path_reuse_sm2.core.state import SkillState
from path_reuse_sm2.core.plugin import skill


@skill("SkillStart")
class SkillStart(SkillState):
    def __init__(self, node, **kwargs):
        super().__init__(node, name="SkillStart")

    def run(self, bb, **kwargs) -> bool:
        self.node.get_logger().info("[SkillStart] Start (dummy)")
        return True
