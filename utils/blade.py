# -*- coding: utf-8 -*-
"""ブレードに働く力を計算する."""

import numpy as np
import math_function as mf
from math import pi

import wingElement as we


def getPositionsOfWingElements(n_elem, root_pos, att, length):
    """直線ブレード内の翼素の位置を計算する.

    n_elem(ents): 翼素の数
    root_pos(ition): ブレードの根本の位置
    att(itude): 機体に対するブレードの姿勢.
        [0,0,0]でブレード軸は機体y軸正方向を向く
    length: ブレードの長さ
    """
    roll = att[0]
    yaw = att[2]
    rotation = mf.getYXZRotationMatrixFrom(roll=roll, yaw=yaw)
    vecBlade = np.dot(rotation, [0.0, 1.0, 0.0])
    ds = np.dot(length/n_elem, vecBlade)

    arrPosWingElems = np.zeros((3, n_elem), dtype=float)
    arrPosWingElems[:, 0] = root_pos
    for i in range(n_elem):
        s_elem = i * ds
        arrPosWingElems[:, i] = root_pos + s_elem

    return arrPosWingElems


def testGetpositionsOfWingElements():
    """テスト用."""
    n_elements = 100
    root_position = [0.0, 1.0, 0.0]
    blade_atittude = [10.0 * pi/180, 0.0, 90.0 * pi/180]
    blade_length = 10.0

    arrPosWingElems = getPositionsOfWingElements(
        n_elements,
        root_position,
        blade_atittude,
        blade_length
    )

    # s = arrPosWingElems.shape
    # if s!=[3, n_elements]
    #     raise RuntimeError("")

    print(arrPosWingElems)


def testRotationSpeed():
    """'wingelement' モジュールと組み合わせて回転速度が正しい値となるかのチェック."""
    n_elements = 10
    root_position = [0.0, -1.0, 0.0]
    blade_atittude = [10.0 * pi/180, 0.0, 180.0 * pi/180]
    blade_length = 10.0

    arrPosWingElems = getPositionsOfWingElements(
        n_elements,
        root_position,
        blade_atittude,
        blade_length
    )
    print(arrPosWingElems)

    pqr = np.array([0.0, 0.0, 1.0])

    arrRotation = np.zeros((3, n_elements), dtype=float)
    for i in range(n_elements):
        pos = arrPosWingElems[:, i]
        arrRotation[:, i] = we.rotationalAirspeed(pos, pqr)
    print(arrRotation)


if __name__ == '__main__':
    # testGetpositionsOfWingElements()
    testRotationSpeed()
