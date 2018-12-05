# -*- coding:utf-8 -*-
"""翼素に働く空気力を計算するためのモジュール."""

import numpy as np
from math import cos, sin, atan2, pi, sqrt
# from airfoils import NACA0012_181127


def getYXZRotationMatrixFrom(roll, pitch, yaw):
    """オイラー角から回転行列を作成する.

    * 回転順序がヨー→ロール→ピッチであることに注意.
    * 上反角を先に指定するため
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


def rotationalSpeed(blade_root, element_pos, wing_attitude, angularVel_body):
    """翼素の回転によって生じる対気速度ベクトルを機体座標系で計算する.

    翼素は機体に固定された極座標系で表現する
    blade_root: ブレードの根本の座標
    element_position: ブレードの根本からの距離
    wing_attitude: 翼素の姿勢(オイラー角)
    """
    rot = getYXZRotationMatrixFrom(0.0, wing_attitude[0], wing_attitude[2])
    bladeDirection = np.dot(rot, [0.0, 1.0, 0.0])
    position = blade_root + np.dot(element_pos, bladeDirection)

    rotationSpeed = - np.cross(position, angularVel_body)
    return rotationSpeed


def flowVelocityForWingElement(flow_body, wing_attitude):
    """機体座標系で表された流速を翼素視点の座標に変換する."""
    wingRoll = wing_attitude[0]
    wingPitch = wing_attitude[1]
    wingYaw = wing_attitude[2]

    rot = getYXZRotationMatrixFrom(wingRoll, wingPitch, wingYaw)
    flow_wing = np.dot(flow_body, rot)
    return flow_wing


def aeroforce(func_airfoil, rho, u):
    """翼型, 翼素固定座標系上の流速, 迎角から翼素に働く圧力を求める."""
    alpha = atan2(u[2], u[0])
    Cl, Cd = func_airfoil(alpha)
    rot = np.array([
        [cos(alpha), sin(alpha)],
        [-sin(alpha), cos(alpha)]
    ])
    pressure = 0.5 * rho * (u[0] ** 2 + u[2]**2) * np.dot(rot, [Cd, Cl])

    return pressure


def test_flowVelocityForWingElement(test_flow, test_attitude, true_attitude):
    """関数が正しい値を出しているかチェック."""
    flow_body = test_flow
    wing_attitude = test_attitude
    flow_wing = flowVelocityForWingElement(flow_body, wing_attitude)
    err = flow_wing - true_attitude
    norm_err = np.dot(err, err)
    print(flow_wing)

    ret = False
    if norm_err < 10E-6:
        ret = True
    return ret


if __name__ == '__main__':
    test_flow = [-1, 0, 0]
    test = [0, 30*pi/180, 0]
    true = [-sqrt(3.0)/2, 0, -0.5]
    print(test_flowVelocityForWingElement(test_flow, test, true))

    test_flow = [-1, 1, 0]
    test = [0, 30*pi/180, 0]
    true = [-sqrt(3.0)/2, 1, -0.5]
    print(test_flowVelocityForWingElement(test_flow, test, true))

    test_flow = [-1, 1, 0]
    test = [0, 30*pi/180, 90*pi/180]
    true = [1, sqrt(3.0)/2, -0.5]
    print(test_flowVelocityForWingElement(test_flow, test, true))
