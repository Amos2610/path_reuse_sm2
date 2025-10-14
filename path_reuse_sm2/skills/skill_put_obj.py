#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Put Object Skill
"""
from path_reuse_sm2.core.state import SkillState
from path_reuse_sm2.core.plugin import skill


@skill("SkillPutObj")
class SkillPutObj(SkillState):
    def __init__(self, node, **kwargs):
        super().__init__(node, name="SkillPutObj")

    def run(self, bb, **kwargs) -> bool:
        self.node.get_logger().info("[SkillPutObj] Put (dummy)")
        return True
