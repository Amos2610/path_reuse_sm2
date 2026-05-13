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
    Find Obj Skill (Dummy implementation for RO-MAN2026 to avoid dependency issues)
    """
    def __init__(self, node, **kwargs):
        super().__init__(outcomes=[SUCCEED, ABORT, CANCEL])
        self.node = node
        self._sm = StateMachine(outcomes=["success", "except"])
        find_obj = FindObj(node, **kwargs)
        self._sm.add_state("FIND_OBJ", find_obj,
                        transitions={"success": "success", "loop": "FIND_OBJ", "except": "except"})   
        self.kwargs = kwargs   

    def set_blackboard(self, blackboard: Dict[str, Any]) -> bool:
        # default values for find object skill
        workpiece = self.kwargs.get("workpiece") or self.kwargs.get("target") or ""
        target = self.kwargs.get("target") or workpiece

        # RAG は "joints"/"pose" で渡してくる。既に blackboard にあれば上書きしない
        joints = self.kwargs.get("joints") or self.kwargs.get("obj_joints") or []
        pose = self.kwargs.get("pose") or self.kwargs.get("grasp_pose") or None

        blackboard["obj_joints"] = joints
        if pose:
            blackboard["grasp_pose"] = pose
        blackboard["workpiece"] = workpiece
        blackboard["target"] = target
        blackboard["object_class"] = self.kwargs.get("object_class", "")
        blackboard["query"] = self.kwargs.get("query", "")

        return True

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillFindObj..")
        if not self.set_blackboard(blackboard):
            return ABORT
        outcome = self._sm(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT