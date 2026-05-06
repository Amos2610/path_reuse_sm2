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
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK


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

    def _bb_get(self, bb, key, default=None):
        try:
            return bb.get(key, default)
        except Exception:
            try:
                return getattr(bb, key)
            except Exception:
                return default

    def _bb_set(self, bb, key, value):
        try:
            bb[key] = value
        except Exception:
            setattr(bb, key, value)

    def _is_joint_list(self, value):
        if not isinstance(value, (list, tuple)):
            return False
        if len(value) < 6:
            return False
        try:
            [float(v) for v in value[:6]]
            return True
        except Exception:
            return False

    def get_blackboard(self, bb) -> bool:
        obj_joints = self._bb_get(bb, "obj_joints")

        if self._is_joint_list(obj_joints):
            return True

        joints_from_rag = self.kwargs.get("joints")
        if self._is_joint_list(joints_from_rag):
            joints = [float(v) for v in joints_from_rag[:6]]
            self.node.get_logger().info(f"[SkillGraspObj] Using joints from RAG args: {joints}")
            self._bb_set(bb, "obj_joints", joints)
            return True

        if obj_joints is not None:
            self.node.get_logger().info("[SkillGraspObj] obj_joints is pose-like. Grasp state will resolve it.")
            return True

        if self.kwargs.get("pose") is not None or self.kwargs.get("grasp_pose") is not None:
            self.node.get_logger().info("[SkillGraspObj] grasp pose exists. Grasp state will compute IK.")
            return True

        self.node.get_logger().error("Blackboard missing 'obj_joints' and no grasp pose fallback.")
        return False

    def execute(self, blackboard: Dict[str, Any]) -> str:
        self.node.get_logger().info("Executing SkillGraspObj...")
        if not self.get_blackboard(blackboard):
            return ABORT
        outcome = self._sm(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT