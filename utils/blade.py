# -*- coding: utf-8 -*-
"""ブレードに働く力を計算する."""

import numpy as np
import math_function as mf
from math import pi

import wingElement as we
import airfoils as af
import math_function as mf


def getPositionsOfWingElements(n_elem, root_pos, att, length):
    """直線ブレード内の翼素の位置を計算する.

    n_elem(ents): 翼素のセクション数
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

    arrPosWingElems = np.zeros((3, n_elem+1), dtype=float)
    arrPosWingElems[:, 0] = root_pos
    for i in range(n_elem+1):
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


def getRotationalAirSpeed(pqr, arrPosWingElems, n_elements):
    """翼素ごとの回転による対気速度(機体座標系)."""
    arrRotation = np.zeros((3, n_elements+1), dtype=float)
    for i in range(n_elements+1):
        pos = arrPosWingElems[:, i]
        arrRotation[:, i] = we.rotationalAirspeed(pos, pqr)
    return arrRotation


def testRotationSpeed():
    """'wingelement' モジュールと組み合わせて回転速度が正しい値となるかのチェック."""
    n_elements = 10
    root_position = [0.0, 0.06, 0.0]
    blade_atittude = [0.0 * pi/180, -80.0* pi/180, 0.0 * pi/180]
    blade_length = 0.13
    chord_length = 0.1

    arrPosWingElems = getPositionsOfWingElements(
        n_elements,
        root_position,
        blade_atittude,
        blade_length
    )
    # print(arrPosWingElems)

    # 翼素ごとの回転による対気速度(機体座標系)
    pqr = np.array([0.0, 0.0, -1.0])
    # arrRotation = np.zeros((3, n_elements+1), dtype=float)
    # for i in range(n_elements+1):
    #     pos = arrPosWingElems[:, i]
    #     arrRotation[:, i] = we.rotationalAirspeed(pos, pqr)
    # print(arrRotation)
    arrRotation = getRotationalAirSpeed(pqr, arrPosWingElems, n_elements)

    # 翼素ごとの対気速度(翼素座標系)
    velocity_body = np.array([0.0, 0.0, 9.8])
    arrAeroVelocity = arrRotation
    for i in range(arrAeroVelocity.shape[1]):
        arrAeroVelocity[:,i] -= velocity_body
        # 翼素座標系に変換
        rot = mf.getYXZRotationMatrixFrom(
            roll = -blade_atittude[0],
            pitch = -blade_atittude[1],
            yaw = -blade_atittude[2]
        )
        arrAeroVelocity[:,i] = np.dot(rot, arrAeroVelocity[:,i])
    print(arrAeroVelocity)

    # 翼素ごとの圧力(翼素座標系)
    listPressure = np.frompyfunc(we.aeroPressure, 5, 1)(af.NACA0012_181127, 1.295, arrAeroVelocity[0,:], arrAeroVelocity[1,:], arrAeroVelocity[2,:])
    arrPressure = np.zeros((3,n_elements+1), dtype=float)
    for i in range(n_elements+1):
        arrPressure[0,i] = listPressure[i][0]
        arrPressure[2,i] = listPressure[i][1]
    print(arrPressure)

    # 翼素ごとの圧力(機体座標系)
    for i in range(arrPressure.shape[1]):
        rot_inv = np.linalg.inv(rot)
        arrPressure[:,i] = np.dot(rot, arrPressure[:,i])
    print(arrPressure)

    # 翼素に働く力と，モーメントを求める
    arrdForce= arrPressure * chord_length * blade_length / n_elements
    arrdMoments = np.zeros((3,n_elements+1), dtype=float)
    for i in range(n_elements+1):
        arrdMoments[:,i] = np.cross(arrPosWingElems[:,i], arrdForce[:,i])

    # 翼素に働く力をシンプソン積分して，ブレード全体の力を求める
    sumF = np.zeros((3),dtype=float)
    sumM = np.zeros((3),dtype=float)
    for i in range(n_elements//2):
        sumF += arrdForce[:,i*2] + 4 * arrdForce[:,i*2+1] + arrdForce[:,i*2+2]
        sumM += arrdMoments[:,i*2] + 4 * arrdMoments[:,i*2+1] + arrdMoments[:,i*2+2]
    print(sumF, sumM)



if __name__ == '__main__':
    # testGetpositionsOfWingElements()
    testRotationSpeed()
