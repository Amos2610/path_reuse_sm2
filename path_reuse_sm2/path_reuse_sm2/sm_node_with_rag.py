#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRSM (Path Reuse-based State Machine) のメインノード。
TaskSet サービスで受け取った Skill[] から flow と StateMachine を構築して、
そのタスク専用の PRSM を 1 回だけ実行する。
"""

from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from path_reuse_sm2_interfaces.srv import TaskSet
from path_reuse_sm2_interfaces.msg import TaskInfo, Skill as SkillMsg

from yasmin.blackboard import Blackboard
from yasmin.state_machine import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
# from yasmin_viewer import YasminViewerPub  # 必要になったら復活

from path_reuse_sm2.core.plugin import discover, get, names
from path_reuse_sm2.skills.skill_standard import SkillStart, SkillRAGBridge


class PRSMNode(Node):
    """
    PRSM の中心となる ROS 2 ノード。
    起動 → スキル発見 → TaskSet サービス待ち → TaskSet で flow/sm 構築 → 実行
    """

    def __init__(self) -> None:
        super().__init__("prsm_node")

        # --- 1) パラメータ定義と読込（ループだけ使う） ---
        self._declare_params()
        self._loop = self.get_parameter("loop").value

        # --- 2) スキルを自動発見（@skill デコレータ登録を有効化）---
        self._discover_skills()

        # --- 3) ステートマシン（まだ構築しない。TaskSet ごとに作る） ---
        self.sm: StateMachine | None = None

        # --- 4) コールバックグループ & TaskSet サービス ---
        self._sm_cb_group = MutuallyExclusiveCallbackGroup()
        self._task_set_srv = self.create_service(
            TaskSet,
            "prsm_task_set",
            self._task_set_callback,
            callback_group=self._sm_cb_group,
        )

        self.get_logger().info("[PRSM] Node initialized. Waiting for /prsm_task_set service calls...")

    # -------------------------
    # パラメータ処理
    # -------------------------
    def _declare_params(self) -> None:
        """本ノードが受け取る ROS パラメータを宣言する。"""
        # start_delay: Viewer購読準備のための起動遅延（秒）
        self.declare_parameter("start_delay", 0.5)
        # loop: flowの最後→最初に戻る（Trueでループ、Falseで一回きり）
        self.declare_parameter("loop", False)
        # use_pathseed: PathSeedを使うかどうか
        self.declare_parameter("use_pathseed", True)
        # pathseed_grasp: Grasp用PathSeedファイルパス
        self.declare_parameter("pathseed_grasp", "")
        # pathseed_put: Put用PathSeedファイルパス
        self.declare_parameter("pathseed_put", "")
        # phase: 現在の動作フェーズ
        # self.declare_parameter("phase", "Initial_Phase")
        self.declare_parameter("grasp_phase", "Initial_Phase")
        self.declare_parameter("put_phase", "Initial_Phase")
        self.declare_parameter("pathseed_registry_path", "")

    # -------------------------
    # TaskSet サービスコールバック
    # -------------------------
    def _task_set_callback(self, request: TaskSet.Request, response: TaskSet.Response):
        available = set(names())
        requested = [s.skill_name for s in request.task.skills]
        unknown = [n for n in requested if n not in available]
        if unknown:
            response.accepted = False
            response.message = f"Unknown skill(s): {unknown}"
            self.get_logger().error(response.message)
            return response

        task: TaskInfo = request.task
        skills = task.skills

        if len(skills) == 0:
            self.get_logger().warn("[PRSM] Received empty task.")
            response.accepted = False
            response.message = "Empty task."
            return response

        self.get_logger().info("[PRSM] 📥 Received TaskSet Request")
        self.get_logger().info(f"[PRSM]   Skill Count: {len(skills)}")

        # 1) TaskSet の Skill[] から flow を作る
        #    [{"name": "SkillFindObj", "args": {...}}, ...] の形に変換
        flow: List[Dict[str, Any]] = []
        for i, s in enumerate(skills):
            pose_frame_id = getattr(s, "pose_frame_id", "")
            step = {
                "name": s.skill_name,
                "args": {
                    "skill_name": s.skill_name,
                    "source_location": s.source_location,
                    "target_location": s.target_location,
                    "workpiece": s.workpiece,
                    "joints": list(s.joints),
                    "pose": list(s.pose),
                    "pose_frame_id": pose_frame_id,
                    "grasp_pose_frame_id": pose_frame_id,
                    "path_seed_path": s.path_seed_path,
                },
            }
            flow.append(step)
            self.get_logger().info(
                f"Skill[{i}] name={repr(s.skill_name)} target={repr(s.target_location)} workpiece={repr(s.workpiece)} path_seed_path={repr(s.path_seed_path)}"
            )
            self.get_logger().info(
                f"[PRSM][frame] Skill[{i}] name={repr(s.skill_name)} "
                f"pose_len={len(s.pose)}, pose_frame_id={repr(pose_frame_id)}"
            )

        # 2) flow に応じてステートマシンを作り直す
        self.get_logger().info("[PRSM] Building state machine...")
        self.sm = self._build_state_machine(flow)
        self.get_logger().info("[PRSM] State machine built.")

        # 3) Blackboard に Skill[] を詰める
        bb = Blackboard()
        bb["skills"] = list(skills)  # Skill.msg の配列をそのまま渡す

        # 4) 実行（このサービス呼び出しの中で 1 回だけ）
        self.get_logger().info("[PRSM] Executing state machine...")
        outcome = self.sm(bb)
        self.get_logger().info(f"[PRSM] outcome: {outcome}")

        response.accepted = True
        response.message = f"Task executed with outcome={outcome}"
        return response

    # -------------------------
    # スキルの発見
    # -------------------------
    def _discover_skills(self) -> None:
        """skills パッケージを探索し、@skill デコレータにより登録させる。"""
        discover("path_reuse_sm2.skills")
        self.get_logger().info(f"[PRSM] skills discovered: {names()}")

    # -------------------------
    # ステートマシン構築
    # -------------------------
    def _build_transitions(self, flow_names: List[str], loop: bool) -> Dict[str, Dict[str, str]]:
        """
        フローの遷移表を作成。
        - state_id = "S{i}_{Name}"
        - 成功(SUCCEED): 次ステートへ。loop=True のとき最後は先頭へ戻す。
        - 失敗(ABORT)  : ABORT 終了。
        - キャンセル(CANCEL): CANCEL 終了。
        """
        transitions: Dict[str, Dict[str, str]] = {}
        n = len(flow_names)
        for i, name in enumerate(flow_names):
            state_id = f"S{i}_{name}"
            if i < n - 1:
                next_id = f"S{i+1}_{flow_names[i+1]}"
            else:
                next_id = "RAGBridge" if loop and n > 0 else SUCCEED
            transitions[state_id] = {
                SUCCEED: next_id,
                ABORT: ABORT,
                CANCEL: CANCEL,
            }
        return transitions

    def _build_state_machine(self, flow: List[Dict[str, Any]]) -> StateMachine:
        """
        flow の順番に沿って Skill(State) を生成・登録。
        ループ指定(self._loop=True)なら、最後の SUCCEED 遷移は先頭に戻す。
        """
        sm = StateMachine(outcomes=[SUCCEED, ABORT, CANCEL])

        # 1) Start state -> Bridge state (RAG check)
        sm.add_state(
            name="START",
            state=SkillStart(node=self),
            transitions={SUCCEED: "RAGBridge" if flow else SUCCEED},
        )

        # 2) Bridge state (RAG check) -> first skill or SUCCEED
        sm.add_state(
            name="RAGBridge",
            state=SkillRAGBridge(node=self),
            transitions={
                SUCCEED: f"S0_{flow[0]['name']}" if flow else SUCCEED,
                ABORT: ABORT,
                CANCEL: CANCEL,
            },
        )

        flow_names = [step["name"] for step in flow]
        transitions_map = self._build_transitions(flow_names, loop=self._loop)

        for i, step in enumerate(flow):
            name = step["name"]
            args = step.get("args", {}) or {}

            try:
                SkillCls = get(name)
            except KeyError:
                self.get_logger().error(f"[PRSM] skill not found: {name}")
                continue

            state_id = f"S{i}_{name}"

            # Skill 側の __init__(node, step_index=..., **kwargs) で受ける想定
            args["step_index"] = i

            self.get_logger().info(
                f"[PRSM] Creating state {state_id} with args keys={list(args.keys())}"
            )
            state = SkillCls(self, **args)
            sm.add_state(state_id, state, transitions=transitions_map[state_id])
            self.get_logger().info(f"[PRSM] Added state {state_id}")

        return sm


def main():
    """
    エントリポイント。
    - rclpy 初期化 → ノード生成 → spin
    - 実行トリガは TaskSet サービスのみ
    """
    rclpy.init()
    node = PRSMNode()
    try:
        exec = MultiThreadedExecutor(num_threads=2)
        exec.add_node(node)
        exec.spin()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
