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
