#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict
from yasmin.state import State
from rclpy.parameter import Parameter
from path_reuse_method.path_seed_client import PathSeedClient
from path_reuse_sm2.core.path_registry import PathRegistry
import os


class UpdatePathSeed(State):
    def __init__(self, node, type: str = "grasp", **kwargs):
        super().__init__(outcomes=["success", "except"])
        self.pr_node = node
        self.type = type
        self.pr_client = None

        # kwargs から情報を取得
        self.source_id = kwargs.get("source_location", "HOME")
        # grasp の場合は workpiece を target として扱う (grasp: CTN_6 -> WP_1 の形)
        # put の場合は target_location を target として扱う (put: WP_1 -> CTN_1 の形)
        if type == "grasp":
            self.target_id = kwargs.get("workpiece") or kwargs.get("target_location") or ""
        else:
            self.target_id = kwargs.get("target_location") or kwargs.get("workpiece") or ""
        # SkillGraspObj / SkillPutObj に合わせる
        default_skill_name = "SkillGraspObj" if type == "grasp" else "SkillPutObj"
        self.action_name = kwargs.get("skill_name") or default_skill_name

        # Path Registry (rag_factory_specific_task_agent の data ディレクトリを指す)
        registry_path = self.pr_node.get_parameter("pathseed_registry_path").value
        if not registry_path.startswith('/'):
            registry_path = os.path.join(self._get_workspace_root(), registry_path)
            
        self.pr_node.get_logger().info(f"[UpdatePathSeed] Action: {self.action_name}, Source: {self.source_id}, Target: {self.target_id}")
        self.pr_node.get_logger().info(f"[UpdatePathSeed] Registry path: {registry_path}")
        self.registry = PathRegistry(registry_path)

    def _get_workspace_root(self) -> str:
        import os
        try:
            from ament_index_python.packages import get_package_prefix
            install_prefix = get_package_prefix('path_reuse_sm2')
            return os.path.dirname(os.path.dirname(install_prefix))
        except Exception:
            ament_prefix = os.environ.get('AMENT_PREFIX_PATH', '')
            if ament_prefix:
                first_path = ament_prefix.split(':')[0]
                return os.path.dirname(os.path.dirname(first_path))
            return os.getcwd()

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
        
        # ex1_pick_and_place/ 部分を取り出して updated フォルダを構築
        try:
            lib_rel = pathseed_path.split("pathseeds/Library/")[1].split("/pre_defined/")[0]
        except IndexError:
            self.pr_node.get_logger().error(f"[UpdatePathSeed] Unexpected pathseed_path format: {pathseed_path}")
            return "except"

        # source_id, target_id からファイル名を生成 (例: update_pick_WP1_CTN2.txt)
        src = self.source_id.replace(" ", "_")
        tgt = self.target_id.replace(" ", "_")
        if self.type == "grasp":
            filename = f"update_pick_{src}_{tgt}.txt"
        elif self.type == "put":
            filename = f"update_place_{src}_{tgt}.txt"
        else:
            self.pr_node.get_logger().error(f"Unknown type: {self.type}")
            return "except"
        
        encode_pathseed_file = f"{lib_rel}/updated/{filename}"
        self.pr_node.get_logger().info(f"[UpdatePathSeed] Saving to: {encode_pathseed_file}")
        # PathSeedClientの呼び出し
        # self, trajectory=None, trajectory_file_path=None, relative_saved_path=None):
        if self.pr_client is None:
            self.pr_node.get_logger().info("[UpdatePathSeed] Creating PathSeedClient for pathseed update.")
            self.pr_client = PathSeedClient()
        success, saved_path = self.pr_client.send_encode_path_seed(
            trajectory=update_trajectory,
            trajectory_file_path="",
            relative_saved_path=encode_pathseed_file,
        )
        if not success:
            self.pr_node.get_logger().error("Failed to update path seed.")
            return "except"

        self.pr_node.get_logger().info(f"Path seed updated successfully: {saved_path}")

        # Path Registry を更新
        self.registry.update(
            source_id=self.source_id,
            target_id=self.target_id,
            action=self.action_name,
            path_seed=encode_pathseed_file
        )
        self.pr_node.get_logger().info(f"[UpdatePathSeed] Registry updated: ({self.source_id} -> {self.target_id}) action={self.action_name}")

        self.pr_node.set_parameters(
            [Parameter(name=f"{self.type}_phase", value="Imprementation_Phase")]
        )
        return "success"
