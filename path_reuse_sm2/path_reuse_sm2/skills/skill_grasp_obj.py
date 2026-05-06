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
        self.kwargs = kwargs
        use_pathseed = self.node.get_parameter("use_pathseed").value
        grasp = Grasp(node, **kwargs)
        update_path_seed = UpdatePathSeed(node, type="grasp", **kwargs)

        self._sm = StateMachine(outcomes=["success", "except"])
        if use_pathseed is True:
            self._sm.add_state("GRASP", grasp,
                            transitions={"success": "UPDATE_PATHSEED", "loop": "GRASP", "except": "except"})
            self._sm.add_state("UPDATE_PATHSEED", update_path_seed,
                            transitions={"success": "success", "except": "except"})
        else:
            self._sm.add_state("GRASP", grasp,
                            transitions={"success": "success", "loop": "GRASP", "except": "except"})
        self._viewer_pub = YasminViewerPub(
            fsm_name="skill_grasp_obj_viewer",
            node=self.node,
            rate=10.0,
            fsm=self._sm
        )

    def get_blackboard(self, bb) -> bool:
        obj_joints = None

        try:
            obj_joints = bb.get("obj_joints")
        except Exception:
            try:
                obj_joints = getattr(bb, "obj_joints")
            except Exception:
                obj_joints = None

        if obj_joints is None:
            joints_from_rag = self.kwargs.get("joints")
            if joints_from_rag and len(joints_from_rag) > 0:
                self.node.get_logger().info(f"[SkillGraspObj] 'obj_joints' missing on BB. Using joints from RAG: {joints_from_rag}")
                try:
                    bb["obj_joints"] = joints_from_rag
                except Exception:
                    setattr(bb, "obj_joints", joints_from_rag)
                return True

            self.node.get_logger().error("Blackboard missing 'obj_joints' and no fallback joints in RAG args.")
            return False

        return True

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillGraspObj...")
        if not self.get_blackboard(blackboard):
            return ABORT
        outcome = self._sm(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT