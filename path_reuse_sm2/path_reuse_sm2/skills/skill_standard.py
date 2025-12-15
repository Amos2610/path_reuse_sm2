#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standard Skill: Put Object
"""
from std_msgs.msg import Bool
from yasmin.state import State
from yasmin_ros.basic_outcomes import SUCCEED, CANCEL, ABORT
from path_reuse_sm2.core.plugin import skill


@skill("SkillStart")
class SkillStart(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED])
        self.node = node

    def execute(self, blackboard):
        self.node.get_logger().info("[SkillStart] Start (dummy)")
        return SUCCEED


@skill("SkillRAGBridge")
class SkillRAGBridge(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED, CANCEL, ABORT])
        self.task_stop = False
        self.node = node
        # self._task_set_srv = self.create_service(
        #     TaskSet,
        #     "prsm_task_set",
        #     self._task_set_callback,
        #     callback_group=self._sm_cb_group,
        # )
        self.task_stop_sub = self.node.create_subscription(
            Bool,
            "/rag_task_stop",
            self.task_stop_callback,
            10,
        )

    def task_stop_callback(self, msg):
        self.node.get_logger().info(f"[SkillRAGBridge] Received /rag_task_stop: {msg.data}")
        self.task_stop = msg.data

    def execute(self, blackboard):
        self.node.get_logger().info("[SkillRAGBridge] RAG Bridge (dummy)")
        if self.task_stop is True:
            self.node.get_logger().info("[SkillRAGBridge] Task stop requested via RAG.")
            return CANCEL
        return SUCCEED