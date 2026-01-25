#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict
from yasmin.state import State
from yasmin.state_machine import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
# from yasmin_viewer import YasminViewerPub
from path_reuse_sm2.core.plugin import skill

from path_reuse_sm2.skills.state.find import FindObj


@skill("SkillFindObj")
class SkillFindObj(State):
    """
    Find Obj Skill
    """
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED, ABORT, CANCEL])
        self.node = node
        find_obj = FindObj(node, **kwargs)
        self._sm = StateMachine(outcomes=["success", "except"])
        self._sm.add_state("FIND_OBJ", find_obj,
                        transitions={"success": "success", "loop": "FIND_OBJ", "except": "except"})        

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillFindObj...")
        outcome = self._sm.execute(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT