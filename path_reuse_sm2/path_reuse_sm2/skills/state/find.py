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
        
        # TODO: ここに認識処理を実装していく

        # Blackboardに認識結果をセット
        self.pr_node.get_logger().info("[FindObj] set obj_pose")
        setattr(blackboard, "obj_pose", obj_pose) #TODO 一旦C空間のままセット
        return "success"
