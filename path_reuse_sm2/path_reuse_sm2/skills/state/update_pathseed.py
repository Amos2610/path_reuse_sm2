#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict
from yasmin.state import State


class UpdatePathSeed(State):
    def __init__(self, node, margin_mm: float = 5.0):
        super().__init__(outcomes=["success", "except"])
        self.node = node
        self.margin_mm = margin_mm

    def execute(self, blackboard=None):
        self.node.get_logger().info(
            f"UpdatePathSeed state executed. margin_mm={self.margin_mm}"
        )
        time.sleep(1.0)  # Simulate processing time
        return "success"
