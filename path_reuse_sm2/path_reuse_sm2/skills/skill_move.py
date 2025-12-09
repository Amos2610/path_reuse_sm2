#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move Skill: Move to target joints or pose
"""
from yasmin.state import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin.state_machine import StateMachine
from path_reuse_sm2.core.plugin import skill

from path_reuse_sm2.skills.state.move import Move


@skill("SkillMove")
class SkillMove(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED])
        self.node = node
        move_state = Move(node)
        self._sm = StateMachine(outcomes=["success", "except"])
        self._sm.add_state("MOVE", move_state,
                        transitions={"success": "success", "except": "except"})

    def execute(self, blackboard):
        self.node.get_logger().info("Executing SkillMove...")
        outcome = self._sm.execute(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT