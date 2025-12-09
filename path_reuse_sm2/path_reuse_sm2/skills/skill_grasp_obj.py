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
from path_reuse_sm2_interfaces.msg import Skill


@skill("SkillGraspObj")
class SkillGraspObj(State):
    """
    Grasp Skill
    """
    def __init__(self, node, step_index: int = 0, **kwargs):
        super().__init__(outcomes=[SUCCEED, ABORT, CANCEL])
        self.node = node
        self.step_index = step_index
        grasp = Grasp(node, approach_margin=kwargs.get("grasp_approach_margin", 0.01))
        update_path_seed = UpdatePathSeed(node, margin_mm=kwargs.get("seed_margin_mm", 5.0), type="grasp")

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
        if not hasattr(bb, "obj_pose") or bb.obj_pose is None:
            self.node.get_logger().error("Blackboard missing 'obj_pose'.")
            return False
        return True        

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillGraspObj...")

        skills = getattr(blackboard, "skills", None)
        if skills is None or self.step_index >= len(skills):
            self.node.get_logger().error("Invalid step_index or missing skills in blackboard.")
            return ABORT
    
        if self.step_index >= len(skills):
            self.node.get_logger().error("step_index out of range of skills.")
            return ABORT
        
        skill_msg: Skill = skills[self.step_index]

        self.node.get_logger().info(
            f"[SkillGraspObj] from TaskSet: "
            f"target={skill_msg.target_location}, "
            f"workpiece={skill_msg.workpiece}, "
            f"path_seed_path={skill_msg.path_seed_path}"
        )
        blackboard.target_location = skill_msg.target_location
        blackboard.workpiece = skill_msg.workpiece
        blackboard.path_seed_path = skill_msg.path_seed_path
        
        if not self.get_blackboard(blackboard):
            return ABORT
        outcome = self._sm.execute(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT