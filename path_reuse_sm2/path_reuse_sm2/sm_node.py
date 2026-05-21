#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PRSM (Path Reuse-based State Machine) のメインノード。
"""
import os
import json
from typing import Any, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# モジュールのインポート（YASMIN関連）
# https://github.com/uleroboticsgroup/yasmin.git
from yasmin.blackboard import Blackboard
from yasmin.state_machine import StateMachine
from yasmin_ros.basic_outcomes import SUCCEED, ABORT, CANCEL
from yasmin_viewer import YasminViewerPub

# 自作（同一パッケージ内）
from path_reuse_sm2.core.plugin import discover, get, names
from path_reuse_sm2.core.flow import normalize
from path_reuse_sm2.skills.skill_standard import SkillStart


class PRSMNode(Node):
    """
    PRSMの中心となるROS 2ノード。
    起動 → パラメータ読込 → スキル発見 → ステートマシン構築 → Viewer連携 → 実行
    """
    def __init__(self) -> None:
        super().__init__("prsm_node")

        # --- 1) パラメータ定義と読込 ---
        self._declare_params()
        raw_flow, flow_args_json, start_delay = self._read_params()

        self._loop = self.get_parameter("loop").value

        # --- 2) Flowの正規化（["SkillA", ...] -> [{"name":"SkillA"}, ...]）---
        self.flow: List[Dict[str, Any]] = self._normalize_flow(raw_flow)

        # --- 3) 追加引数のマージ（flow_args_jsonを各step.argsへ）---
        self._merge_extra_args(self.flow, flow_args_json)

        # --- 5) スキルを自動発見（@skillデコレータ登録を有効化）---
        self._discover_skills()

        # --- 6) ステートマシン構築（SkillStateをそのまま登録）---
        self.sm = self._build_state_machine(self.flow)

        # --- 7) Viewer連携（状態遷移をブラウザ表示）---
        self._viewer_pub = YasminViewerPub(
            fsm_name="prsm_viewer",
            fsm=self.sm,
            rate=10.0,
            node=self
        )
        self.get_logger().info("[PRSM] Viewer attached (signature: topic,node,rate,state_machine).")

        # --- 8) 実行
        self._sm_cb_group = MutuallyExclusiveCallbackGroup() # ステートマシン用のCallbackGroup
        self._start_timer = self.create_timer(
            start_delay, self._run_once_and_stop_timer, callback_group=self._sm_cb_group
        )
        self.get_logger().info(f"[PRSM] initialized. start_delay={start_delay} sec")

    # -------------------------
    # パラメータ処理
    # -------------------------
    def _declare_params(self) -> None:
        """本ノードが受け取るROSパラメータを宣言する。"""
        # flow: ["SkillGraspObj", ...] または [{"name": "SkillGraspObj", "args": {...}}, ...]
        # self.declare_parameter("flow", ["SkillMove", "SkillFindObj", "SkillGraspObj", "SkillPutObj"])
        self.declare_parameter("flow", ["SkillMove" "SkillFindObj"])
        # flow_args_json: {"SkillGraspObj": {"speed": 0.5}, "SkillPutObj": {"place": "binA"}} のような追加引数
        self.declare_parameter("flow_args_json", "")
        # start_delay: Viewer購読準備のための起動遅延（秒）
        self.declare_parameter("start_delay", 0.5)
        # loop: flowの最後→最初に戻る（Trueでループ、Falseで一回きり）
        self.declare_parameter("loop", False)
        # use_pathseed: PathSeedを使うかどうか
        self.declare_parameter("use_pathseed", False)
        # pathseed_grasp: Grasp用PathSeedファイルパス
        self.declare_parameter("pathseed_grasp", "")
        # pathseed_put: Put用PathSeedファイルパス
        self.declare_parameter("pathseed_put", "")
        # phase: 現在の動作フェーズ
        # self.declare_parameter("phase", "Initial_Phase")
        self.declare_parameter("grasp_phase", "Initial_Phase")
        self.declare_parameter("put_phase", "Initial_Phase")
        self.declare_parameter("pathseed_registry_path", "")
        self.declare_parameter("grasp_require_approval", True)
        self.declare_parameter("grasp_approval_timeout_sec", 0.0)

    def _read_params(self):
        """宣言済みのパラメータ値を取得して返す。"""
        raw_flow = self.get_parameter("flow").value
        flow_args_json = self.get_parameter("flow_args_json").value
        start_delay = float(self.get_parameter("start_delay").value)
        return raw_flow, flow_args_json, start_delay

    # -------------------------
    # Flow処理
    # -------------------------
    def _normalize_flow(self, raw_flow: Any) -> List[Dict[str, Any]]:
        """ユーザが与えたflowを正規化して[{name:str, args:dict}, ...]に統一する。"""
        flow = normalize(raw_flow)
        self.get_logger().info(f"[PRSM] flow(normalized): {flow}")
        return flow

    def _merge_extra_args(self, flow: List[Dict[str, Any]], flow_args_json: str) -> None:
        """flow_args_jsonの内容を、同名Skillのargsへマージする（任意）。"""
        if not isinstance(flow_args_json, str) or not flow_args_json.strip():
            return

        try:
            extra = json.loads(flow_args_json)
        except Exception as e:
            self.get_logger().warn(f"[PRSM] flow_args_json parse error: {e}")
            return

        if not isinstance(extra, dict):
            self.get_logger().warn("[PRSM] flow_args_json must be an object (dict). Ignored.")
            return

        for step in flow:
            name = step.get("name")
            if not name:
                continue
            if "args" not in step or not isinstance(step["args"], dict):
                step["args"] = {}
            step["args"].update(extra.get(name, {}))

        self.get_logger().info(f"[PRSM] flow(after extra args): {flow}")

    # -------------------------
    # パラメータファイル
    # -------------------------
    def _log_params_file(self, params_file: str) -> None:
        """params_fileの存在確認だけ行い、ログに出す（将来ここで読み込み）。"""
        if isinstance(params_file, str) and params_file:
            exists = os.path.exists(params_file)
            self.get_logger().info(f"[PRSM] params_file: {params_file} "
                                   f"{'(exists)' if exists else '(missing)'}")

    # -------------------------
    # スキルの発見
    # -------------------------
    def _discover_skills(self) -> None:
        """skillsパッケージを探索し、@skillデコレータにより登録させる。"""
        discover("path_reuse_sm2.skills")
        self.get_logger().info(f"[PRSM] skills discovered: {names()}")

    # -------------------------
    # ステートマシン構築
    # -------------------------
    def _build_transitions(self, flow_names: List[str], loop: bool) -> Dict[str, Dict[str, str]]:
        """
        フローの遷移表を作成。
        - state_id = "S{i}_{Name}"
        - 成功(SUCCEED): 次ステートへ。loop=Trueのとき最後は先頭へ戻す。
        - 失敗(ABORT)  : ABORT 終了。
        """
        transitions: Dict[str, Dict[str, str]] = {}
        n = len(flow_names)
        for i, name in enumerate(flow_names):
            state_id = f"S{i}_{name}"
            if i < n - 1:
                next_id = f"S{i+1}_{flow_names[i+1]}"
            else:
                next_id = f"S0_{flow_names[0]}" if loop and n > 0 else SUCCEED
            transitions[state_id] = {SUCCEED: next_id, ABORT: ABORT}
        return transitions

    def _build_state_machine(self, flow: List[Dict[str, Any]]) -> StateMachine:
        """
        flowの順番に沿ってSkill(State)を生成・登録。
        ループ指定(self._loop=True)なら、最後のSUCCEED遷移は先頭に戻す。
        """
        sm = StateMachine(outcomes=[SUCCEED, ABORT, CANCEL])

        sm.add_state(
            name="START",
            state=SkillStart(node=self),
            transitions={SUCCEED: f"S0_{flow[0]['name']}" if flow else SUCCEED}
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
                # 未登録スキルは飛ばす（必要ならダミーStateを足してABORTでも可）
                continue

            state_id = f"S{i}_{name}"
            state = SkillCls(self, **args)  # SkillState(State) 継承
            sm.add_state(state_id, state, transitions=transitions_map[state_id])

        return sm

    # -------------------------
    # 実行
    # -------------------------
    def _run_once_and_stop_timer(self) -> None:
        """
        一回だけステートマシンを実行して、結果をログ出力する。
        - Viewer購読準備のための短い遅延後に呼ばれる（start_delay秒）
        - 実行後、hold_afterがFalseならノードとrclpyを終了する
        """
        # このTimerは一回きりで良いので、最初に止めておく
        self._start_timer.cancel()

        bb = Blackboard()  # Blackboardはステート間で共有するデータ置き場
        outcome = self.sm(bb)
        self.get_logger().info(f"[PRSM] outcome: {outcome}")


def main():
    """
    エントリポイント。
    - rclpy初期化 → ノード生成 → spin（実行はノード側のTimerで一回だけ）
    """
    rclpy.init()
    node = PRSMNode()
    try:
        exec = MultiThreadedExecutor(num_threads=2)
        exec.add_node(node)
        exec.spin()
        # rclpy.spin(node)
    finally:
        # hold_after=Trueの場合、Ctrl+Cなどで抜けるとここに来る
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
