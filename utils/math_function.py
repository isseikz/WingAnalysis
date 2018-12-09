# -*- coding:utf-8 -*-
"""数学関数をまとめたモジュール."""

import numpy as np
from math import sin, cos


def getYXZRotationMatrixFrom(roll=0.0, pitch=0.0, yaw=0.0):
    """オイラー角から回転行列を作成する.

    * 回転順序がヨー→ロール→ピッチであることに注意.
    """
    Ryaw = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])
    Rpitch = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])
    Rroll = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])
    rotationMatrix = np.dot(Rpitch, np.dot(Rroll, Ryaw))
    return rotationMatrix


def euler2Quartanion(roll, pitch, yaw):
    """オイラー角をクオータニオンに変換する."""
    cosR_2 = cos(roll/2)
    sinR_2 = sin(roll/2)
    cosP_2 = cos(pitch/2)
    sinP_2 = sin(pitch/2)
    cosY_2 = cos(yaw/2)
    sinY_2 = sin(yaw/2)

    q0 = cosR_2 * cosP_2 * cosY_2 + sinR_2 * sinP_2 * sinY_2
    q1 = sinR_2 * cosP_2 * cosY_2 - cosR_2 * sinP_2 * sinY_2
    q2 = cosR_2 * sinP_2 * cosY_2 + sinR_2 * cosP_2 * sinY_2
    q3 = cosR_2 * cosP_2 * sinY_2 - sinR_2 * sinP_2 * cosY_2

    return np.array([q0, q1, q2, q3])
