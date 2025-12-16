#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any
from yasmin.state import State
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper


class FindObj(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=["success", "except", "loop"])
        XArmUtilsWrapper.__init__(self)
        self.pr_node = node
        self.pr_client = PathSeedClient()
        self.target_location = kwargs.get("target_location", "")
        self.workpiece = kwargs.get("workpiece", "")
        self.path_seed_path = kwargs.get("path_seed_path", "")
        self.step_index = kwargs.get("step_index", 0)

    def execute(self, blackboard: Any = None) -> str:
        self.pr_node.get_logger().info("FindObj state executed.")
        obj_joints = [0.916, 0.724, -1.70014, 0.001, 0.977, -0.67]  # TODO: 仮の値
        
        # TODO: ここに認識処理を実装していく
        # self.workpieceを認識してobj_jointsを取得する

        # Blackboardに認識結果をセット
        self.pr_node.get_logger().info("[FindObj] set obj_joints")
        setattr(blackboard, "obj_joints", obj_joints) #TODO 一旦C空間のままセット
        return "success"
