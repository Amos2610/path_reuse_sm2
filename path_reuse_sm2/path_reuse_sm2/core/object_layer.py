#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""掴んだ物体を planning scene に attach / detach するためのクライアント。

掴んでいる間、プランナが「その物体はロボットの一部」として扱えるようにする。
これをやらないと、運んでいるワークが障害物判定から抜け落ちる。逆に detach を
忘れると、放したあともロボットが物体を抱えている前提で計画され続ける。

対向は octomap_workspace_recognition2 の object_layer_action_server が提供する
``/attach_object`` と ``/detach_object``（型は octomap_workspace_recognition2_interfaces）。

AttachObject は形状を呼び出し側から受け取る。PRSM は認識由来の形状を持たないことが
多いので、blackboard にあればそれを使い、無ければパラメータの代替箱を手先リンクの
座標系に置く（`resolve_shape`）。

**PRSM ノードとは別の rclpy ノード**を持ち `spin_until_future_complete` で同期呼び出しする。
ステートマシンは `/prsm_task_set` のコールバック内で同期実行されるため、prsm_node 自身の
クライアントを同期呼び出しすると入れ子 spin でデッドロックする。
"""

import math
from typing import Optional, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose
from shape_msgs.msg import Mesh, SolidPrimitive

from octomap_workspace_recognition2_interfaces.action import AttachObject, DetachObject


_DEFAULT_ATTACH_ACTION = "/attach_object"
_DEFAULT_DETACH_ACTION = "/detach_object"
_FALLBACK_OBJECT_ID = "prsm_workpiece"


def _param(pr_node, name: str, default):
    """PRSM ノードのパラメータを取りに行く。未宣言でも落とさない。"""
    try:
        value = pr_node.get_parameter(name).value
    except Exception:
        return default
    return default if value is None else value


class ObjectLayerClient:
    """`/attach_object` と `/detach_object` を叩くだけのクライアント（プロセス内シングルトン）。"""

    _instance = None

    def __init__(self, attach_action: str, detach_action: str) -> None:
        self.attach_action = attach_action
        self.detach_action = detach_action
        self._node = Node("prsm_object_layer_client")
        self._attach = ActionClient(self._node, AttachObject, attach_action)
        self._detach = ActionClient(self._node, DetachObject, detach_action)

    @classmethod
    def get(cls, attach_action: str, detach_action: str) -> "ObjectLayerClient":
        if (
            cls._instance is None
            or cls._instance.attach_action != attach_action
            or cls._instance.detach_action != detach_action
        ):
            cls._instance = ObjectLayerClient(attach_action, detach_action)
        return cls._instance

    def send(self, client: ActionClient, goal, wait_timeout_s: float, call_timeout_s: float):
        """(result, error) を返す。結果が得られれば error は None。"""
        if not client.wait_for_server(timeout_sec=wait_timeout_s):
            return None, f"action server not available: {client._action_name}"

        goal_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, goal_future, timeout_sec=call_timeout_s)
        goal_handle = goal_future.result()
        if goal_handle is None:
            return None, f"goal was not accepted within {call_timeout_s}s"
        if not goal_handle.accepted:
            return None, "goal rejected by the server"

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=call_timeout_s)
        wrapped = result_future.result()
        if wrapped is None:
            return None, f"result did not arrive within {call_timeout_s}s"
        return wrapped.result, None


def _mesh_defect(mesh: Mesh) -> str:
    """FCL に載せられないメッシュなら理由を返す。健全なら空文字。"""
    n = len(mesh.vertices)
    for i, v in enumerate(mesh.vertices):
        if not all(math.isfinite(c) for c in (v.x, v.y, v.z)):
            return f"頂点 {i} が非有限 ({v.x}, {v.y}, {v.z})"
    degenerate = 0
    for k, t in enumerate(mesh.triangles):
        idx = list(t.vertex_indices)
        if len(idx) != 3 or any(i < 0 or i >= n for i in idx):
            return f"三角形 {k} の頂点番号 {idx} が範囲外（頂点数 {n}）"
        if len(set(idx)) < 3:
            degenerate += 1
            continue
        a, b, c = (mesh.vertices[i] for i in idx)
        ux, uy, uz = b.x - a.x, b.y - a.y, b.z - a.z
        vx, vy, vz = c.x - a.x, c.y - a.y, c.z - a.z
        cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        if (cx * cx + cy * cy + cz * cz) < 1e-16:
            degenerate += 1
    if mesh.triangles and degenerate == len(mesh.triangles):
        return f"三角形 {len(mesh.triangles)} 個すべてが退化"
    return ""


def _aabb_size(mesh: Mesh):
    """有限頂点の AABB の辺長。頂点が無ければ None。"""
    pts = [(v.x, v.y, v.z) for v in mesh.vertices if all(math.isfinite(c) for c in (v.x, v.y, v.z))]
    if not pts:
        return None
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return [hi[i] - lo[i] for i in range(3)]


def resolve_shape(
    pr_node,
    primitive: Optional[SolidPrimitive] = None,
    pose: Optional[Pose] = None,
    frame_id: str = "",
    mesh: Optional[Mesh] = None,
) -> Tuple[Optional[Mesh], SolidPrimitive, Pose, str]:
    """attach する形状・姿勢・座標系を決める。優先順位は mesh > primitive > 代替箱。

    揃っていなければ代替箱を手先リンクの座標系に置く。リンク座標系に置くと TF を
    引かずに済み、「見た時刻の TF ではなく最新の TF で変換して誤った位置に付く」
    問題を構造的に避けられる。代替箱に落ちたら WARN を出す（当たり判定が実物と
    違うことをログで追えるように）。

    戻り値は `(mesh|None, primitive, pose, frame_id)`。mesh を使う場合でも primitive は
    常に非 None で返し、goal には両方載せる。
    """
    link = str(_param(pr_node, "end_effector_attach_link", "link_tcp"))
    has_mesh = mesh is not None and len(mesh.vertices) >= 4 and len(mesh.triangles) >= 4
    mesh_box: Optional[SolidPrimitive] = None
    if has_mesh:
        bad = _mesh_defect(mesh)
        if bad:
            # 壊れたメッシュを attach すると move_group が FCL の自己干渉チェックで
            # 落ちることがある。世界物体としては通っても attach で初めて FCL に載るので
            # ここで弾き、AABB の箱で代用する
            pr_node.get_logger().warn(
                f"[ObjectLayer] 認識由来のメッシュが壊れている（{bad}）ので attach しない。"
                "メッシュの AABB の箱で代用する"
            )
            size = _aabb_size(mesh)
            if size is not None and all(sz > 1e-3 for sz in size):
                mesh_box = SolidPrimitive()
                mesh_box.type = SolidPrimitive.BOX
                mesh_box.dimensions = [float(v) for v in size]
            has_mesh = False

    def _fallback_box() -> SolidPrimitive:
        size = list(_param(pr_node, "end_effector_object_size", [0.05, 0.05, 0.05]))
        if len(size) != 3:
            pr_node.get_logger().warn(
                f"[ObjectLayer] end_effector_object_size は 3 要素で指定する（今の値: {size}）。既定値に落とす"
            )
            size = [0.05, 0.05, 0.05]
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [float(v) for v in size]
        return box

    def _placeholder_pose() -> Pose:
        placeholder = Pose()
        placeholder.position.z = float(_param(pr_node, "end_effector_object_offset_z", 0.0))
        placeholder.orientation.w = 1.0
        return placeholder

    box = primitive if primitive is not None else (mesh_box if mesh_box is not None else _fallback_box())
    if mesh_box is not None and primitive is None:
        primitive = mesh_box

    if (has_mesh or primitive is not None) and pose is not None and frame_id:
        return (mesh if has_mesh else None), box, pose, frame_id

    if has_mesh or primitive is not None:
        # 形だけ分かっている。掴んでいる以上、物は手先にあるはずなので姿勢は代替値
        return (mesh if has_mesh else None), box, _placeholder_pose(), link

    pr_node.get_logger().warn(
        f"[ObjectLayer] 認識由来の形状が無いので代替箱 {list(box.dimensions)} を attach する。"
        "掴んだ物の当たり判定は実物と違う"
    )
    return None, box, _placeholder_pose(), link


def attach_object(
    pr_node,
    object_id: str,
    *,
    primitive: Optional[SolidPrimitive] = None,
    pose: Optional[Pose] = None,
    frame_id: str = "",
    mesh: Optional[Mesh] = None,
    tag: str = "PRSM",
) -> bool:
    """掴んだ物体を planning scene へ attach する。

    simulate_only でも呼ぶ。planning scene の操作でしかなく物理的な副作用が無い上、
    続く Put の計画が「持っている状態」で行われる（simulate 後は sm_node が巻き戻す）。
    失敗は既定では警告に留めて True を返す。`end_effector_attach_required` が真のときだけ
    False を返す。
    """
    if not bool(_param(pr_node, "end_effector_attach_object", True)):
        return True

    required = bool(_param(pr_node, "end_effector_attach_required", False))
    object_id = (object_id or "").strip() or _FALLBACK_OBJECT_ID

    shape_mesh, shape_box, shape_pose, shape_frame = resolve_shape(pr_node, primitive, pose, frame_id, mesh)

    goal = AttachObject.Goal()
    goal.request_id = f"prsm-{tag}"
    goal.object_id = object_id
    goal.frame_id = shape_frame
    goal.mesh = shape_mesh if shape_mesh is not None else Mesh()
    goal.primitive = shape_box
    goal.pose = shape_pose
    goal.attach_link = str(_param(pr_node, "end_effector_attach_link", ""))

    return _run(
        pr_node,
        lambda client: (client._attach, goal),
        what=(
            f"attach {object_id} ({shape_frame}) "
            + (f"mesh({len(goal.mesh.vertices)}v/{len(goal.mesh.triangles)}t)" if shape_mesh is not None
               else f"box{[round(d, 4) for d in shape_box.dimensions]}")
        ),
        required=required,
        tag=tag,
    )


def detach_object(
    pr_node,
    object_id: str,
    *,
    remove_from_scene: bool = False,
    tag: str = "PRSM",
) -> bool:
    """attach していた物体を放す。remove_from_scene が偽なら置いた場所に物体が残る。"""
    if not bool(_param(pr_node, "end_effector_attach_object", True)):
        return True

    required = bool(_param(pr_node, "end_effector_attach_required", False))
    object_id = (object_id or "").strip() or _FALLBACK_OBJECT_ID

    goal = DetachObject.Goal()
    goal.request_id = f"prsm-{tag}"
    goal.object_id = object_id
    goal.remove_from_scene = bool(remove_from_scene)

    return _run(pr_node, lambda client: (client._detach, goal), what=f"detach {object_id}", required=required, tag=tag)


def _run(pr_node, pick, *, what: str, required: bool, tag: str) -> bool:
    """action を 1 回投げて結果をログに落とす。失敗の扱いは required 次第。"""
    attach_action = str(_param(pr_node, "end_effector_attach_action", _DEFAULT_ATTACH_ACTION))
    detach_action = str(_param(pr_node, "end_effector_detach_action", _DEFAULT_DETACH_ACTION))
    timeout = float(_param(pr_node, "end_effector_attach_timeout_sec", 5.0))

    try:
        client = ObjectLayerClient.get(attach_action, detach_action)
        action_client, goal = pick(client)
        result, err = client.send(action_client, goal, timeout, timeout)
    except Exception as e:  # noqa: BLE001 - シーン操作の失敗でタスクを落とさない
        pr_node.get_logger().warn(f"[{tag}] {what} failed: {e}")
        return not required

    if result is None:
        pr_node.get_logger().warn(f"[{tag}] {what} failed: {err}")
        return not required

    if not result.success:
        pr_node.get_logger().warn(f"[{tag}] {what} failed: {result.message}")
        return not required

    pr_node.get_logger().info(f"[{tag}] {what} ok ({result.message})")
    return True
