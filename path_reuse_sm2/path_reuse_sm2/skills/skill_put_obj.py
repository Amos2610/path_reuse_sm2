#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Put Object Skill
"""
from typing import Any, Dict
from yasmin.state import State
from yasmin.state_machine import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
# from yasmin_viewer import YasminViewerPub
from path_reuse_sm2.core.plugin import skill

from path_reuse_sm2.skills.state.put import Put
from path_reuse_sm2.skills.state.update_pathseed import UpdatePathSeed


@skill("SkillPutObj")
class SkillPutObj(State):
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED, ABORT, CANCEL])
        self.node = node
        put = Put(node, approach_margin=kwargs.get("put_approach_margin", 0.01))
        update_path_seed = UpdatePathSeed(node, margin_mm=kwargs.get("seed_margin_mm", 5.0), type="put")

        self._sm = StateMachine(outcomes=["success", "except"])
        self._sm.add_state("PUT", put,
                        transitions={"success": "UPDATE_PATHSEED", "loop": "PUT", "except": "except"})
        self._sm.add_state("UPDATE_PATHSEED", update_path_seed,
                        transitions={"success": "success", "except": "except"})
        
        # self._viewer_pub = YasminViewerPub(
        #     fsm_name="skill_put_obj_viewer",
        #     node=self.node,
        #     rate=10.0,
        #     fsm=self._sm
        # )

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillPutObj...")
        outcome = self._sm.execute(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT