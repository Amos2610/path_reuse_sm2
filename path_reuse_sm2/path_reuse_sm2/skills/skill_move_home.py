#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from yasmin.state import State
from yasmin.state_machine import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
from path_reuse_sm2.core.plugin import skill
from path_reuse_sm2.core.xarm_utils import XArmUtilsWrapper
from path_reuse_sm2.skills.state.move import Move


@skill("SkillMoveHome")
class SkillMoveHome(XArmUtilsWrapper, State):
    """
    ホームポジションへ戻るスキル。
    グリッパーを開いてからホームへ移動する。
    simulate_only モードではグリッパー操作をスキップする。
    """

    def __init__(self, node, **kwargs):
        State.__init__(self, outcomes=[SUCCEED, ABORT, CANCEL])
        XArmUtilsWrapper.__init__(self)
        self.node = node
        move_state = Move(node, **{**kwargs, "skill_name": "SkillMoveHome"})
        self._sm = StateMachine(outcomes=["success", "except", "loop"])
        self._sm.add_state(
            "MOVE", move_state,
            transitions={"success": "success", "except": "except", "loop": "MOVE"},
        )

    def execute(self, blackboard):
        self.node.get_logger().info("Executing SkillMoveHome...")
        simulate_only = self.node.get_parameter("prsm_simulate_only").value
        if not simulate_only:
            try:
                self.xarm.gripper_open()
            except Exception as e:
                self.node.get_logger().warn(f"[SkillMoveHome] gripper_open failed but continue: {e}")
        outcome = self._sm(blackboard)
        if outcome == "success":
            return SUCCEED
        else:
            return ABORT
