#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRSM (Path Reuse-based State Machine) のメインノード。
TaskSet サービスで受け取った Skill[] から flow と StateMachine を構築して、
そのタスク専用の PRSM を 1 回だけ実行する。

監視パラメータにより、外部ノード（実行監視ノード等）から
各スキルの実行状況をリアルタイムに取得可能。
"""

from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from path_reuse_sm2_interfaces.srv import TaskSet
from path_reuse_sm2_interfaces.msg import TaskInfo, Skill as SkillMsg

from yasmin.blackboard import Blackboard
from yasmin.state import State
from yasmin.state_machine import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
# from yasmin_viewer import YasminViewerPub  # 必要になったら復活

from path_reuse_sm2.core.plugin import discover, get, names
from path_reuse_sm2.skills.skill_standard import SkillStart, SkillRAGBridge


class MonitoredSkillState(State):
    """
    既存の SkillState をラップして、execute() 前後に
    監視用パラメータを自動更新するプロキシ State。
    これにより各スキルの実行開始・完了が外部から追跡可能になる。
    """

    def __init__(self, node: Node, inner_state: State, skill_name: str, skill_index: int):
        # inner_state と同じ outcomes を持つ
        super().__init__(outcomes=list(inner_state.get_outcomes()))
        self._node = node
        self._inner = inner_state
        self._skill_name = skill_name
        self._skill_index = skill_index

    def execute(self, blackboard) -> str:
        """inner_state の execute() をラップし、前後でパラメータを更新する。"""
        # --- 実行前: 現在のスキル情報を更新 ---
        self._node.get_logger().info(
            f"[PRSM Monitor] >>> Entering skill [{self._skill_index}]: {self._skill_name}"
        )
        self._node._update_monitoring_params(
            prsm_current_skill=self._skill_name,
            prsm_current_skill_index=self._skill_index
        )

        # --- 実行 ---
        outcome = self._inner.execute(blackboard)

        # --- 実行後: 結果をログ ---
        self._node.get_logger().info(
            f"[PRSM Monitor] <<< Skill [{self._skill_index}]: {self._skill_name} -> {outcome}"
        )

        return outcome


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

        from std_msgs.msg import String
        self._monitor_pub = self.create_publisher(
            String,
            '/prsm/skill_status',
            100
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

        # --- 監視用パラメータ ---
        # 実行監視ノードがポーリングして、ロボットの現在状態を把握するためのパラメータ群
        self.declare_parameter("prsm_status", "idle")                # idle / running / succeeded / failed / cancelled
        self.declare_parameter("prsm_current_skill", "")             # 現在実行中のスキル名
        self.declare_parameter("prsm_current_skill_index", 0)        # 現在のスキルインデックス
        self.declare_parameter("prsm_total_skills", 0)               # スキル総数
        self.declare_parameter("prsm_outcome", "")                   # 最終結果: succeed / abort / cancel
        self.declare_parameter("prsm_error_message", "")             # エラー詳細

    def _update_monitoring_params(self, **kwargs) -> None:
        """監視用パラメータを一括更新し、同時にTopicでも配信する。"""
        # --- 1) パラメータ更新 (後方互換性のため残す) ---
        params = []
        type_map = {
            str: Parameter.Type.STRING,
            int: Parameter.Type.INTEGER,
            bool: Parameter.Type.BOOL,
        }
        for key, value in kwargs.items():
            ptype = type_map.get(type(value), Parameter.Type.STRING)
            params.append(Parameter(key, ptype, value))
        if params:
            self.set_parameters(params)
            
        # --- 2) Topicによるリアルタイム通知 ---
        import json
        from std_msgs.msg import String
        
        # 既存の状態を読み込みつつ新しい状態をマージする
        current_status = {
            "prsm_status": self.get_parameter("prsm_status").value,
            "prsm_current_skill": self.get_parameter("prsm_current_skill").value,
            "prsm_current_skill_index": self.get_parameter("prsm_current_skill_index").value,
            "prsm_total_skills": self.get_parameter("prsm_total_skills").value,
            "prsm_outcome": self.get_parameter("prsm_outcome").value,
            "prsm_error_message": self.get_parameter("prsm_error_message").value,
        }
        current_status.update(kwargs)
        
        if hasattr(self, '_monitor_pub'):
            msg = String(data=json.dumps(current_status))
            self._monitor_pub.publish(msg)

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
            # 監視パラメータ: エラー状態
            self._update_monitoring_params(
                prsm_status="failed",
                prsm_error_message=response.message,
            )
            return response

        task: TaskInfo = request.task
        skills = task.skills

        if len(skills) == 0:
            self.get_logger().warn("[PRSM] Received empty task.")
            response.accepted = False
            response.message = "Empty task."
            self._update_monitoring_params(
                prsm_status="failed",
                prsm_error_message=response.message,
            )
            return response

        self.get_logger().info("[PRSM] 📥 Received TaskSet Request")
        self.get_logger().info(f"[PRSM]   Skill Count: {len(skills)}")

        # --- 監視パラメータ: 実行開始 ---
        self._update_monitoring_params(
            prsm_status="running",
            prsm_current_skill="",
            prsm_current_skill_index=0,
            prsm_total_skills=len(skills),
            prsm_outcome="",
            prsm_error_message="",
        )

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
        try:
            outcome = self.sm(bb)
            self.get_logger().info(f"[PRSM] outcome: {outcome}")

            # --- 監視パラメータ: 実行完了 ---
            if outcome == SUCCEED:
                final_status = "succeeded"
            elif outcome == CANCEL:
                final_status = "cancelled"
            else:
                final_status = "failed"

            self._update_monitoring_params(
                prsm_status=final_status,
                prsm_outcome=outcome,
            )

            response.accepted = True
            response.message = f"Task executed with outcome={outcome}"
        except Exception as e:
            self.get_logger().error(f"[PRSM] Execution error: {e}")
            self._update_monitoring_params(
                prsm_status="failed",
                prsm_outcome="abort",
                prsm_error_message=str(e),
            )
            response.accepted = False
            response.message = f"Execution error: {e}"

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
        各スキルは MonitoredSkillState でラップされ、
        execute() 前後に監視用パラメータが自動更新される。
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

            inner_state = SkillCls(self, **args)
            # MonitoredSkillState でラップし、パラメータを自動更新する
            monitored_state = MonitoredSkillState(
                node=self,
                inner_state=inner_state,
                skill_name=name,
                skill_index=i,
            )
            sm.add_state(state_id, monitored_state, transitions=transitions_map[state_id])

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
