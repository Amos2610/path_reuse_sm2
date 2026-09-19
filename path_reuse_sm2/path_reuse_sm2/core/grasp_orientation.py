"""把持姿勢と接近方向の変換。

xArm6 の link_tcp は局所 +Z が接近方向。真下固定姿勢 [1, 0, 0, 0]（X 軸まわり 180 度）は
局所 +Z をワールド -Z に写すので、``approach_from_orientation([1,0,0,0])`` は (0, 0, -1) を
返す。pre-grasp を接近方向の手前に取ると、真下把持では「+Z に上げる」と一致する。
"""


def approach_from_orientation(quat_xyzw):
    """姿勢 [x, y, z, w] で TCP が向く接近方向（局所 +Z のワールド表現）を返す。

    戻り値: 単位ベクトル [x, y, z]。姿勢が壊れていれば None。
    """
    import numpy as np

    x, y, z, w = (float(v) for v in quat_xyzw)
    # 回転行列の 3 列目（局所 +Z）
    local_z = np.array([
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    ])
    norm = np.linalg.norm(local_z)
    if norm < 1e-9:
        return None
    return (local_z / norm).tolist()


def rotate_about_approach(quat_xyzw, approach_xyz, angle_rad):
    """姿勢を接近軸まわりに ``angle_rad`` だけ回す（ベースフレームでの回転を左から掛ける）。

    接近方向は変わらない。IK の逃げ道として使う。平行グリッパでは 180 度以外は
    把持方向が変わるので、どの角度を許すかは呼び手（パラメータ）が決める。
    引数の ``approach_xyz`` は ``quat_xyzw`` と同じフレーム。零ベクトルなら None。
    """
    import numpy as np

    a = np.asarray(approach_xyz, dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return None
    a = a / norm

    half = 0.5 * float(angle_rad)
    s = np.sin(half)
    q_axis = np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(half)])
    q = np.asarray(quat_xyzw, dtype=float)

    x1, y1, z1, w1 = q_axis
    x2, y2, z2, w2 = q
    return [
        float(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2),
        float(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2),
        float(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2),
        float(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2),
    ]
