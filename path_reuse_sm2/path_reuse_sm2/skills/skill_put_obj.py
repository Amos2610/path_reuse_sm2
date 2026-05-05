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
        self.kwargs = kwargs
        use_pathseed = self.node.get_parameter("use_pathseed").value
        put = Put(node, **kwargs)
        update_path_seed = UpdatePathSeed(node, type="put", **kwargs)

        self._sm = StateMachine(outcomes=["success", "except"])
        if use_pathseed is True:
            self._sm.add_state("PUT", put,
                            transitions={"success": "UPDATE_PATHSEED", "loop": "PUT", "except": "except"})
            self._sm.add_state("UPDATE_PATHSEED", update_path_seed,
                            transitions={"success": "success", "except": "except"})
        else:
            self._sm.add_state("PUT", put,
                            transitions={"success": "success", "loop": "PUT", "except": "except"})
        # self._viewer_pub = YasminViewerPub(
        #     fsm_name="skill_put_obj_viewer",
        #     node=self.node,
        #     rate=10.0,
        #     fsm=self._sm
        # )

    def get_blackboard(self, bb) -> bool:
        if not hasattr(bb, "obj_joints") or bb.obj_joints is None:
            # RAG から直接 joints が送られているか確認
            joints_from_rag = self.kwargs.get("joints")
            if joints_from_rag and len(joints_from_rag) > 0:
                self.node.get_logger().info(f"[SkillPutObj] 'obj_joints' missing on BB. Using joints from RAG: {joints_from_rag}")
                setattr(bb, "obj_joints", joints_from_rag)
                return True
            
            self.node.get_logger().error("Blackboard missing 'obj_joints' and no fallback joints in RAG args.")
            return False
        return True        

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillPutObj...")
        if not self.get_blackboard(blackboard):
            return ABORT
        outcome = self._sm(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT