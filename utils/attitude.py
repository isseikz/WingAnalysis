# -*- coding: utf-8 -*-
"""６自由度機体シミュレーション用のクラス."""

import numpy as np
from numpy.linalg import inv
from math import sin, cos, pi, sqrt
import math_function as mf


class Attitude6DoF(object):
    """docstring for Quartanion.

    6自由度の機体の姿勢をクォータニオンで表現するためのクラス
    """

    def __init__(self):
        """初期化."""
        super(Attitude6DoF, self).__init__()
        self.velocityBody = np.array([0.0, 0.0, 0.0])
        self.quartanion = np.array([1.0, 0.0, 0.0, 0.0])
        # [q0, q1, q2, q3] q_hat = cos(theta/2) + vec_n * sin(theta / 2)
        self.omegaBody = np.array([0.0, 0.0, 0.0])
        self.momentOfInertia = np.array([
            [0.001, 0,         0],
            [0, 0.0001,        0],
            [0, 0, 0.00020633122]
        ])
        self.weight = 0.1
        self.position = np.array([0.0, 0.0, 0.0])

    def setQuartanionFrom(self, roll, pitch, yaw):
        """オイラー角からクオータニオンをセットする."""
        self.quartanion = mf.euler2Quartanion(roll, pitch, yaw)
        return self.quartanion

    def rotationOfPositionVector(self, r):
        """位置ベクトルを回転クオータニオンに基づき回転させる.

        クオータニオンが機体の姿勢変化を表しているなら、機体座標上の位置ベクトルを慣性座標系の位置ベクトルに変換する
        """
        if len(r) != 3:
            raise RuntimeError("Inputted vector must be three dimentional.")

        q = self.quartanion

        A11 = q[0]**2+q[1]**2-q[2]**2-q[3]**2
        A12 = 2*(q[1]*q[2]-q[0]*q[3])
        A13 = 2*(q[1]*q[3]+q[0]*q[2])

        A21 = 2*(q[1]*q[2]+q[0]*q[3])
        A22 = q[0]**2-q[1]**2+q[2]**2-q[3]**2
        A23 = 2*(q[2]*q[3]-q[0]*q[1])

        A31 = 2*(q[1]*q[3]-q[0]*q[2])
        A32 = 2*(q[2]*q[3]+q[0]*q[1])
        A33 = q[0]**2-q[1]**2-q[2]**2+q[3]**2

        A = np.array([
            [A11, A12, A13],
            [A21, A22, A23],
            [A31, A32, A33]
        ])
        rRotated = np.dot(A, r)

        return rRotated

    def bodyVector2InertialVector(self, r):
        """機体座標系上のベクトルを慣性座標系の要素に分解する."""
        return self.rotationOfPositionVector(r)

    def inertialVector2BodyVector(self, r):
        """慣性系上のベクトルを機体座標系で表す."""
        if len(r) != 3:
            raise RuntimeError("Position vector must be three dimentional.")

        q = self.quartanion

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        A = np.array([
            [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
            [2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)],
            [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2]
        ])
        rRotated = np.dot(A, r)

        return rRotated

    def calcDerivativeOfQuartanion(self, omega_inertial):
        """位置ベクトルが角速度ω（慣性系）で回転するときのクオータニオンの時間微分."""
        if len(omega_inertial) != 3:
            raise RuntimeError("Angular velocity must be three dimentional.")

        w = omega_inertial
        q = self.quartanion
        w_hat = 0.5 * np.array([
            [0,    -w[0], -w[1], -w[2]],
            [w[0],     0, -w[2],  w[1]],
            [w[1],  w[2],     0, -w[0]],
            [w[2], -w[1],  w[0],     0]
        ])
        qDot = np.dot(w_hat, q)

        return qDot

    def normOfQuartanion(self):
        """クオータニオンのノルムを計算する.

        * 回転クオータニオンの定義上、このノルムは常に1である.
        """
        q = self.quartanion
        norm = sqrt(q[0]**2+q[1]**2+q[2]**2+q[3]**2)
        return norm

    def updateQuartanion(self, omega_body, dt):
        """機体座標系の角速度から、t+dt秒のクオータニオンを計算する."""
        omega_inertial = self.rotationOfPositionVector(omega_body)
        qDot = self.derivativeOfQuartanion(omega_inertial)

        self.quartanion += np.dot(dt, qDot)         # 積分計算
        self.quartanion /= self.normOfQuartanion()  # 正規化
        return self.quartanion

    def updateQuartanionODE(self, quartanion):
        """Scipy odeで求めたquartanionをもちいて 更新."""
        self.quartanion = quartanion
        self.quartanion /= self.normOfQuartanion()  # 正規化
        return self.quartanion

    def calcDerivativeOfOmegaBody(self, moment_body):
        """機体座標系の角速度の微分をモーメントと現在の状態から求める.

        式は『航空機力学入門』（加藤） (1.20)式より
        """
        w = self.omegaBody
        I_body = self.momentOfInertia
        I_inv = inv(I_body)
        M = moment_body
        h = np.dot(I_body, w)  # 角運動量

        dwdt = np.dot(I_inv, (M - np.cross(w, h)))
        print(f"CDO: {M}, {np.cross(w, h)}")
        return dwdt

    def updateOmegaBody(self, dt, moment_body):
        """機体座標系の角速度をモーメントから更新する."""
        self.omegaBody += np.dot(dt, self.derivativeOfOmegaBody(moment_body))
        return self.omegaBody

    # TODO: 並進運動の運動方程式
    def gravityBody(self):
        """現在の姿勢から, 重力を機体座標系の成分に分解する."""
        g_inertial = np.array([0.0, 0.0, 9.81])
        g_body = self.inertialVector2BodyVector(g_inertial)
        return g_body * self.weight

    def calcDerivativeOfVelocityBody(self, force_body):
        """機体座標系の力から機体座標系上の機体速度の時間微分を求める."""
        w = self.omegaBody
        vc = self.velocityBody
        F = force_body
        m = self.weight

        dvcdt = np.dot(F, 1/m) - np.cross(w, vc)
        return dvcdt

    def getVelocityInertial(self):
        """慣性系の速度を取得する."""
        vel_inertial = self.bodyVector2InertialVector(self.velocityBody)
        return vel_inertial


def testRotation(omega):
    """クラスの動作テスト."""
    att = Attitude6DoF()
    nx = np.zeros(3, dtype=float)
    ny = np.zeros(3, dtype=float)
    nz = np.zeros(3, dtype=float)
    for i in range(1000):
        att.updateQuartanion(omega, 1/1000)
        nx = att.rotationOfPositionVector(np.array([1, 0, 0]))
        ny = att.rotationOfPositionVector(np.array([0, 1, 0]))
        nz = att.rotationOfPositionVector(np.array([0, 0, 1]))
    return nx, ny, nz


def testinertialVector2BodyVector():
    """ベクトルを機体座標系で正しく表現できているかのテスト."""
    psi = pi/3
    the = pi/6
    rot = np.array([
        [cos(-psi), -sin(-psi), 0],
        [sin(-psi),  cos(-psi), 0],
        [0, 0, 1]
    ])
    xi = np.array([cos(the), sin(the), 0.0])
    xb_answer = np.dot(rot, xi)

    att = Attitude6DoF()

    test_q = [cos(psi/2), 0.0*sin(psi/2), 0.0*sin(psi/2), 1.0*sin(psi/2)]
    att.quartanion = np.array(test_q)
    xb_test = att.inertialVector2BodyVector(xi)

    print(f'error: {xb_test},{xb_answer}')
    print(f'error: {np.cross(xb_test,xb_answer)}')


if __name__ == '__main__':
    print(testRotation([2*pi, 0, 0]))
    print(testRotation([0, 2*pi, 0]))
    print(testRotation([0, 0, 2*pi]))
    print(testRotation([2*pi, 2*pi, 0]))
    print(testRotation([0, 2*pi, 2*pi]))
    print(testRotation([2*pi, 0, 2*pi]))
    print(testRotation([2*pi, 2*pi, 2*pi]))
    print(testinertialVector2BodyVector())
