# -*- coding: utf-8 -*-
"""ブレードに働く力を計算する."""

import numpy as np
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


def getAirSpeed(uvw, arrRotation, blade_attitude):
    """翼素ごとの対気速度(翼素座標系)."""
    arrAeroVelocity = arrRotation
    for i in range(arrAeroVelocity.shape[1]):
        arrAeroVelocity[:, i] -= uvw
        # 翼素座標系に変換
        rot = mf.getYXZRotationMatrixFrom(
            roll=-blade_attitude[0],
            pitch=-blade_attitude[1],
            yaw=-blade_attitude[2]
        )
        arrAeroVelocity[:, i] = np.dot(rot, arrAeroVelocity[:, i])
    return arrAeroVelocity


def getPressureInWingElem(arrAeroVelocity, blade_attitude, airfoil):
    """翼素ごとの圧力(機体座標系)."""
    # 翼素ごとの圧力(翼素座標系)
    listPressure = np.frompyfunc(we.aeroPressure, 5, 1)(
        airfoil,
        1.295,
        arrAeroVelocity[0, :],
        arrAeroVelocity[1, :],
        arrAeroVelocity[2, :]
    )

    arrPressure = np.zeros((3, arrAeroVelocity.shape[1]), dtype=float)
    for i in range(arrAeroVelocity.shape[1]):
        arrPressure[0, i] = listPressure[i][0]
        arrPressure[2, i] = listPressure[i][1]

    rot = mf.getYXZRotationMatrixFrom(
        roll=-blade_attitude[0],
        pitch=-blade_attitude[1],
        yaw=-blade_attitude[2]
    )

    # 翼素ごとの圧力(機体座標系)
    for i in range(arrPressure.shape[1]):
        rot_inv = np.linalg.inv(rot)
        arrPressure[:, i] = np.dot(rot_inv, arrPressure[:, i])
    return arrPressure


def getBladeForceMoment(arrP, arrPosWE, chord_len, blade_len, n_elem):
    """ブレード全体の力を求める.

    arrP(ressure): それぞれの翼素に働く圧力の分布
    arrP(osition)W(ing)E(lements): それぞれも翼素の座標
    chord_len(gth): コード長
    blade_len(gth): ブレードの翼幅
    n_elem(emts): 翼素のセクション数
    """
    # 翼素に働く力と，モーメントを求める
    arrdF = arrP * chord_len * blade_len / n_elem
    arrdM = np.zeros((3, n_elem+1), dtype=float)
    for i in range(n_elem+1):
        arrdM[:, i] = np.cross(arrPosWE[:, i], arrdF[:, i])

    # 翼素に働く力をシンプソン積分して，ブレード全体の力を求める
    sumF = np.zeros((3), dtype=float)
    sumM = np.zeros((3), dtype=float)
    for i in range(n_elem//2):
        sumF += arrdF[:, i*2] + 4 * arrdF[:, i*2+1] + arrdF[:, i*2+2]
        sumM += arrdM[:, i*2] + 4 * arrdM[:, i*2+1] + arrdM[:, i*2+2]
    return sumF/3, sumM/3


class Blade(object):
    """docstring for Blade."""

    def __init__(self, n_elem, pos_root, att, b_len, c_len, airfoil):
        """初期化."""
        super(Blade, self).__init__()
        self.n_elements = n_elem
        self.root_position = pos_root
        self.blade_attitude = att
        self.blade_length = b_len
        self.chord_length = c_len
        self.airfoil = airfoil
        self.arrPosWingElems = getPositionsOfWingElements(
            self.n_elements,
            self.root_position,
            self.blade_attitude,
            self.blade_length
        )


def tutorial_one_blade():
    """ある設定のブレードを持つ機体(u,v,w,p,q,r)に働く力・モーメントを求めるデモ."""
    # 翼の設定
    n_elements = 10
    root_position = [0.0, 0.06, 0.0]
    blade_atittude = [0.0 * pi / 180, -80.0 * pi/180, 0.0 * pi/180]
    blade_length = 0.13
    chord_length = 0.1
    airfoil = af.NACA0012_181127

    arrPosWingElems = getPositionsOfWingElements(
        n_elements,
        root_position,
        blade_atittude,
        blade_length
    )

    # 機体の運動(仮設定)
    pqr = np.array([0.0, 0.0, -1*2*pi])
    uvw = np.array([0.0, 0.0, 9.8])

    # 翼素ごとの回転による対気速度(機体座標系)
    arrRotation = getRotationalAirSpeed(
        pqr,
        arrPosWingElems,
        n_elements
    )
    print('\r\n  --- AirSpeed only from the rotation ---\r\n', arrRotation)

    # 翼素ごとの対気速度(翼素座標系)
    arrAeroVelocity = getAirSpeed(
        uvw,
        arrRotation,
        blade_atittude
    )
    print('\r\n  --- AirSpeed (Wing Frame) --- \r\n', arrAeroVelocity)

    # 翼素ごとの圧力(機体座標系)
    arrPressure = getPressureInWingElem(
        arrAeroVelocity,
        blade_atittude,
        airfoil
    )
    print('\r\n  --- Pressure distribution (Body Frame) --- \r\n', arrPressure)

    sumF, sumM = getBladeForceMoment(
        arrPressure,
        arrPosWingElems,
        chord_length,
        blade_length,
        n_elements
    )
    print('\r\n  --- F and M of the blade (Body Frame) ---\r\n', sumF, sumM)


if __name__ == '__main__':
    # testGetpositionsOfWingElements()
    tutorial_one_blade()
