#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move Skill: Move to target joints or pose
"""
from yasmin.state import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
from yasmin.state_machine import StateMachine
from path_reuse_sm2.core.plugin import skill

from path_reuse_sm2.skills.state.move import Move


@skill(names=["SkillMove", "SkillMoveHome"])
class SkillMove(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED, ABORT, CANCEL])
        self.node = node
        move_state = Move(node, **kwargs)
        self._sm = StateMachine(outcomes=["success", "except", "loop"])
        self._sm.add_state("MOVE", move_state,
                        transitions={"success": "success", "except": "except", "loop": "MOVE"})
        self.node.get_logger().info(f"[SkillMove] Initialized with kwargs: {kwargs}")
        # [SkillMove] Initialized with kwargs: {'target_location': 'desk', 'workpiece': '', 'joints': [3.1415, 0.0, -1.5708, 0.0, 1.5708, 0.0], 'pose': [0.5, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0], 'path_seed_path': '', 'step_index': 0}

    def execute(self, blackboard):
        self.node.get_logger().info("Executing SkillMove...")
        outcome = self._sm(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT