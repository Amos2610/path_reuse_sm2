#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict
from path_reuse_sm2.core.state import SkillState


class Start(SkillState):
    def __init__(self, node):
        super().__init__(node, name="Start", outcomes=["success"])
    def run(self, bb: Dict[str, Any], **_) -> str:
        self.node.get_logger().info("Start state executed.")
        return "success"


class End(SkillState):
    def __init__(self, node):
        super().__init__(node, name="End", outcomes=["success"])
    def run(self, bb: Dict[str, Any], **_) -> str:
        self.node.get_logger().info("End state executed.")
        return "success"