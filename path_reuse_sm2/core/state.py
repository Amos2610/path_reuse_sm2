#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class is a base class for all Skills.
This uses to define a unified state for each skill.
Each Skill should inherit from this class and implement the run method.

このクラスはすべてのスキルの基底クラス
各スキルのステートを統一的に定義するために使用．
各スキルはこのクラスを継承し、runメソッドを実装する必要がある
"""
import time
from yasmin.state import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT


class SkillState(State):
    def __init__(self, node, name=None, outcomes=None, delay=1.0):
        super().__init__(outcomes or [SUCCEED, ABORT])
        self.node = node
        self.name = name or self.__class__.__name__
        self.delay = delay  # delay [sec]

    # デフォルトは run(bb)->bool を想定
    def run(self, bb, **kwargs) -> bool:
        raise NotImplementedError

    # 必要なら各Skill側でこのexecute自体をoverrideしてよい（自由度確保）
    def execute(self, bb):
        try:
            ok = self.run(bb)
            time.sleep(self.delay)  # 必要なら遅延
            return SUCCEED if ok else ABORT
        except Exception as e:
            self.node.get_logger().error(f"[{self.name}] {e}")
            return ABORT