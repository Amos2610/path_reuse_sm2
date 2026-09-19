"""計画の前後処理（計画時間・開始状態・simulate の終端）の共通ヘルパ。

ロボアプリ版 grasp.py の _apply_planning_time / _apply_start_state（440a92b :154-167, 846-866）と
sm_node_with_rag.py の get_sim_start_joints / set_sim_end_joints（:410-415）を、研究版では
Blackboard（TaskSet ごとに新規作成）に持たせる形にまとめたもの。
"""

SIM_END_KEY = "sim_end_joints"


def param(pr_node, name: str, default):
    """PRSM ノードのパラメータを取りに行く。未宣言でも落とさない。"""
    try:
        value = pr_node.get_parameter(name).value
    except Exception:
        return default
    return default if value is None else value


def apply_planning_time(xarm, pr_node, seconds: float, tag: str) -> None:
    """MoveGroupInterface の計画時間を設定する。

    既定は MoveGroupInterface の 5 秒で、これまでどこからも変えられていなかった
    （planning_time_default は宣言のみ。planning_time は /move_group のパラメータではないので
    set_move_group_parameter では届かない）。xarm_utils_cpp に set_planning_time が無い旧ビルド
    では WARN を出して既定のまま。
    """
    if not hasattr(xarm, "set_planning_time"):
        pr_node.get_logger().warn(
            f"[{tag}] xarm_utils_py has no set_planning_time; using MoveGroupInterface default"
        )
        return
    xarm.set_planning_time(float(seconds))
    pr_node.get_logger().info(f"[{tag}] planning time: {float(seconds):.1f}s")


def sim_end_joints(blackboard):
    """直前スキルが simulate で到達した関節値（無ければ None）。"""
    if blackboard is None:
        return None
    try:
        v = blackboard[SIM_END_KEY] if SIM_END_KEY in blackboard else None
    except Exception:
        v = getattr(blackboard, SIM_END_KEY, None)
    return list(v) if v else None


def set_sim_end_joints(blackboard, joints) -> None:
    """simulate で計画した軌道の終端を次のスキルの開始状態として残す。"""
    if blackboard is None or not joints:
        return
    try:
        blackboard[SIM_END_KEY] = list(joints)
    except Exception:
        try:
            setattr(blackboard, SIM_END_KEY, list(joints))
        except Exception:
            pass


def apply_start_state(xarm, pr_node, simulate_only: bool, start_joints, tag: str) -> None:
    """plan() の開始状態を明示的に決める。

    simulate_only では実機が動かないので、直前スキルの終端（start_joints）から計画しないと
    Grasp → Put が繋がらない。実行時は必ず現在状態へ戻す（シングルトンの
    MoveGroupInterface に古い開始状態を残さないため）。
    """
    try:
        if simulate_only and start_joints:
            if not xarm.set_start_state(list(start_joints)):
                pr_node.get_logger().warn(f"[{tag}] set_start_state failed; planning from current state.")
        else:
            xarm.set_start_state_to_current()
    except AttributeError:
        # xarm_utils_cpp が set_start_state 未対応（旧ビルド）
        pr_node.get_logger().warn(
            f"[{tag}] set_start_state is unavailable in xarm_utils_py; planning from current state."
        )
