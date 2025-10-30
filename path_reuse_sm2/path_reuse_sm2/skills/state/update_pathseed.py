#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict
from yasmin.state import State
from rclpy.parameter import Parameter
from path_reuse_method.path_seed_client import PathSeedClient


class UpdatePathSeed(State):
    def __init__(self, node, margin_mm: float = 5.0, type: str = "example"):
        super().__init__(outcomes=["success", "except"])
        self.pr_node = node
        self.type = type
        self.pr_client = PathSeedClient()

    def execute(self, blackboard=None):
        self.pr_node.get_logger().info(
            f"UpdatePathSeed state executed."
        )
        # bb の取得
        if blackboard is not None:
            update_trajectory = getattr(blackboard, f'{self.type}_trajectory', None)
            if update_trajectory is None:
                self.pr_node.get_logger().error(
                    f"Blackboard missing '{self.type}_trajectory' for UpdatePathSeed."
                )
                return "except"
            
        pathseed_path = self.pr_node.get_parameter(f"pathseed_{self.type}").value
        self.pr_node.get_logger().info(f"Path seed path: {pathseed_path}")
        if self.type == "grasp":
            encode_pathseed_file = pathseed_path.split("pathseeds/Library/")[1].split("/pre_defined/")[0] + "/updated/pathseed_pick.txt"
        elif self.type == "put":
            encode_pathseed_file = pathseed_path.split("pathseeds/Library/")[1].split("/pre_defined/")[0] + "/updated/pathseed_place.txt"
        else:
            self.pr_node.get_logger().error(f"Unknown type: {self.type}")
            return "except"
        # PathSeedClientの呼び出し
        # self, trajectory=None, trajectory_file_path=None, relative_saved_path=None):
        success = self.pr_client.send_encode_path_seed(
            trajectory=update_trajectory,
            trajectory_file_path="",
            relative_saved_path=encode_pathseed_file,
        )
        if not success:
            self.pr_node.get_logger().error("Failed to update path seed.")
            return "except"

        self.pr_node.get_logger().info("Path seed updated successfully.")
        self.pr_node.set_parameters(
            [Parameter(name=f"{self.type}_phase", value="Imprementation_Phase")]
        )
        return "success"
