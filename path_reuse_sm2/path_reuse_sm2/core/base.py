#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This class  is a base class for all Skills.
Each Skill should inherit from this class and implement the run method.

このクラスはすべてのスキルの基底クラスです。
各スキルはこのクラスを継承し、runメソッドを実装する必要があります。`
""" 

from __future__ import annotations
from abc import ABC, abstractmethod
from rclpy.node import Node
from yasmin.blackboard import Blackboard


class Skill(ABC):
    """各スキルの共通IF。必要なら run 内部でミニ状態機械を組んでOK。"""
    name: str = "Skill"

    def __init__(self, node: Node) -> None:
        self.node = node

    @abstractmethod
    def run(self, *, bb: Blackboard, **kwargs) -> bool:
        """成功 True / 失敗 False"""
        ...
