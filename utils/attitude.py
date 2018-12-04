import numpy as np
from numpy.linalg import inv
from math import sin, cos, pi, sqrt


class Attitude6DoF(object):
    """docstring for Quartanion.

    6自由度の機体の姿勢をクォータニオンで表現するためのクラス
    """

    def __init__(self):
        """初期化."""
        super(Attitude6DoF, self).__init__()
        self.quartanion = np.array([1.0, 0.0, 0.0, 0.0])
        # [q0, q1, q2, q3] q_hat = cos(theta/2) + vec_n * sin(theta / 2)
        self.derivativeOfQuartanion

    def rotationOfPositionVector(self, r):
        """位置ベクトルを回転クオータニオンに基づき回転させる.

        クオータニオンが機体の姿勢変化を表しているなら、機体座標上の位置ベクトルを慣性座標系の位置ベクトルに変換する
        """
        if len(r) != 3:
            raise RuntimeError("Position vector must be three dimentional.")

        q = self.quartanion
        A = np.array([
            [q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]),         2*(q[1]*q[3]+q[0]*q[2])        ],
            [2*(q[1]*q[2]+q[0]*q[3]),         q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])        ],
            [2*(q[1]*q[3]-q[0]*q[2]),         2*(q[2]*q[3]+q[0]*q[1]),         q[0]**2-q[1]**2-q[2]**2+q[3]**2]
        ])
        rRotated = np.dot(A, r)

        return rRotated

    def inertialVectorInBodyFrame(self, r):
        """慣性系上の位置ベクトルを機体座標系で表す。"""
        if len(r)!=3:
            raise RuntimeError("Position vector must be three dimentional.")

        q = self.quartanion
        A = np.array([
            [q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]+q[0]*q[3]),         2*(q[1]*q[3]-q[0]*q[2])        ],
            [2*(q[1]*q[2]-q[0]*q[3]),         q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]+q[0]*q[1])        ],
            [2*(q[1]*q[3]+q[0]*q[2]),         2*(q[2]*q[3]-q[0]*q[1]),         q[0]**2-q[1]**2-q[2]**2+q[3]**2]
        ])
        rRotated = np.dot(A,r)

        return rRotated

    def derivativeOfQuartanion(self, omega_inertial):
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

    def derivativeOfOmegaBody(self, moment_body):
        """機体座標系の角速度の微分をモーメントと現在の状態から求める.

        式は『航空機力学入門』（加藤） (1.20)式より
        """
        w = self.omegaBody
        I = self.momentOfInertia
        I_inv = inv(I)
        M = moment_body
        h = np.dot(I, w)  # 角運動量

        dwdt = np.dot(I_inv, (M - np.cross(w, h)))  # TODO: 妥当性チェック
        return dwdt

    def updateOmegaBody(self, moment_body):
        """機体座標系の角速度をモーメントから更新する."""
        pass


def testRotation(omega):
    """クラスの動作テスト"""
    att = Attitude6DoF()
    nx=np.zeros(3,dtype=float)
    ny=np.zeros(3,dtype=float)
    nz=np.zeros(3,dtype=float)
    for i in range(1000):
        att.updateQuartanion(omega,1/1000)
        nx = att.rotationOfPositionVector(np.array([1,0,0]))
        ny = att.rotationOfPositionVector(np.array([0,1,0]))
        nz = att.rotationOfPositionVector(np.array([0,0,1]))
    return nx,ny,nz


def testInertialVectorInBodyFrame():
    """ベクトルを機体座標系で正しく表現できているかのテスト."""
    psi = pi/3
    the = pi/6
    rot = np.array([
        [cos(-psi), -sin(-psi), 0],
        [sin(-psi),  cos(-psi), 0],
        [        0,          0, 1]
    ])
    xi = np.array([cos(the), sin(the), 0.0])
    xb_answer = np.dot(rot, xi)

    att = Attitude6DoF()
    att.quartanion = [cos(psi/2), 0.0*sin(psi/2), 0.0*sin(psi/2), 1.0*sin(psi/2)]
    xb_test = att.inertialVectorInBodyFrame(xi)

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
    print(testInertialVectorInBodyFrame())
