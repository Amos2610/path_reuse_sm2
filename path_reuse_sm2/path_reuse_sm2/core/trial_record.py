"""試行（TaskSet 1 件）の記録: どの段階で・どの検査で止まったか。

実験の主表「入力カテゴリ × 停止段階」の元データ。検証機構ではなく計測なので、
挙動は変えずに記録だけ足す。

停止段階ラベル（実行境界側）:
  E-accept  受付で拒否（未知スキル・空タスク・状態機械の構築失敗）
  E-seed    パスシードが無い／不整合（decode 空、要求時に無し）
  E-ik      逆運動学・関節限界（開始／目標の関節値が解けない）
  E-col     計画失敗（実行前の衝突検査・計画タイムアウト。MoveIt の error_code を添える）
  E-abort   状態機械の中断・取消（原因を特定できない except）
  EXEC      実行に到達（success=True なら完走、False なら実行中の失敗）
"""
import datetime as _dt
import json
import os
import threading
import time

_LOCK = threading.Lock()
_COUNTER = {"n": 0}


class TrialRecord:
    def __init__(self, simulate_only: bool, skills):
        with _LOCK:
            _COUNTER["n"] += 1
            n = _COUNTER["n"]
        now = _dt.datetime.now()
        self.trial_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{n:04d}"
        self.started_at = now.isoformat(timespec="milliseconds")
        self._t0 = time.monotonic()
        self.simulate_only = bool(simulate_only)
        self.skills = list(skills)
        self.events = []
        self.stage = None
        self.check = None
        self.detail = None
        self.success = None
        self.outcome = None
        self.finished_at = None
        self.elapsed_sec = None

    def event(self, kind: str, **fields):
        e = {"t": round(time.monotonic() - self._t0, 3), "kind": kind}
        e.update(fields)
        self.events.append(e)

    def stop(self, stage: str, check: str, detail: str = ""):
        """最初に止まった段階だけを残す（後から上書きしない）。"""
        self.event("stop", stage=stage, check=check, detail=detail)
        if self.stage is None:
            self.stage, self.check, self.detail = stage, check, detail

    def finish(self, outcome: str):
        self.outcome = outcome
        self.finished_at = _dt.datetime.now().isoformat(timespec="milliseconds")
        self.elapsed_sec = round(time.monotonic() - self._t0, 3)
        if self.stage is None:
            if outcome in ("succeeded", "success"):
                self.stage, self.check, self.detail, self.success = "EXEC", "completed", "", True
            else:
                self.stage, self.check, self.detail = "E-abort", "state_machine", f"outcome={outcome}"
        if self.success is None:
            self.success = self.stage == "EXEC" and self.check == "completed"

    def summary(self) -> dict:
        return {
            "prsm_trial_id": self.trial_id,
            "prsm_stage": self.stage or "",
            "prsm_check": self.check or "",
            "prsm_detail": self.detail or "",
        }

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_sec": self.elapsed_sec,
            "simulate_only": self.simulate_only,
            "skills": self.skills,
            "stage": self.stage,
            "check": self.check,
            "detail": self.detail,
            "success": self.success,
            "outcome": self.outcome,
            "events": self.events,
        }

    def write(self, log_dir: str) -> str:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{self.trial_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path


def current(node):
    """ノードが持つ進行中の TrialRecord（無ければ None）。"""
    return getattr(node, "_trial", None)


def record_event(node, kind: str, **fields) -> None:
    tr = current(node)
    if tr is not None:
        tr.event(kind, **fields)


def record_stop(node, stage: str, check: str, detail: str = "") -> None:
    tr = current(node)
    if tr is not None:
        tr.stop(stage, check, detail)
