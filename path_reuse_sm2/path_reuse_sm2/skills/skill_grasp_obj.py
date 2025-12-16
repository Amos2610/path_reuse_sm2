#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict
from yasmin.state import State
from yasmin.state_machine import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
from yasmin_viewer import YasminViewerPub
from path_reuse_sm2.core.plugin import skill

from path_reuse_sm2.skills.state.grasp import Grasp
from path_reuse_sm2.skills.state.update_pathseed import UpdatePathSeed


@skill("SkillGraspObj")
class SkillGraspObj(State):
    """
    Grasp Skill
    """
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED, ABORT, CANCEL])
        self.node = node
        grasp = Grasp(node, **kwargs)
        update_path_seed = UpdatePathSeed(node, type="grasp", **kwargs)

        self._sm = StateMachine(outcomes=["success", "except"])
        self._sm.add_state("GRASP", grasp,
                        transitions={"success": "UPDATE_PATHSEED", "loop": "GRASP", "except": "except"})
        self._sm.add_state("UPDATE_PATHSEED", update_path_seed,
                        transitions={"success": "success", "except": "except"})
        
        self._viewer_pub = YasminViewerPub(
            fsm_name="skill_grasp_obj_viewer",
            node=self.node,
            rate=10.0,
            fsm=self._sm
        )

    def get_blackboard(self, bb) -> bool:
        if not hasattr(bb, "obj_joints") or bb.obj_joints is None:
            self.node.get_logger().error("Blackboard missing 'obj_joints'.")
            return False
        return True        

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillGraspObj...")
        if not self.get_blackboard(blackboard):
            return ABORT
        outcome = self._sm.execute(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT