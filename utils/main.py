"""3次元シミュレーション."""

import numpy as np
from scipy.integrate import ode
from math import pi

import blade
import airfoils as af
import attitude


def blade_FM(x, blades, disp=True):
    """翼に働く力を出力する."""
    # q = x[0:4]
    pqr = x[4:7]
    # xyz = x[7:10]
    uvw = x[10:13]

    n_blade = len(blades)
    arrFM = np.zeros((n_blade, 2, 3), dtype=float)
    sumF = np.zeros(3, dtype=float)
    sumM = np.zeros(3, dtype=float)

    for id, b in enumerate(blades):
        if disp:
            print(f'\r\n --- {id}_wing calculation ---')

        # 翼素ごとの回転による対気速度(機体座標系)
        arrRotation = blade.getRotationalAirSpeed(
            pqr,
            b.arrPosWingElems,
            b.n_elements
        )
        if disp:
            print(
                '\r\n  --- AirSpeed only from the rotation ---\r\n',
                arrRotation
            )

        # 翼素ごとの対気速度(翼素座標系)
        arrAeroVelocity = blade.getAirSpeed(
            uvw,
            arrRotation,
            b.blade_attitude
        )
        if disp:
            print(
                '\r\n  --- AirSpeed (Wing Frame) --- \r\n',
                arrAeroVelocity
            )

        # 翼素ごとの圧力(機体座標系)
        arrPressure = blade.getPressureInWingElem(
            arrAeroVelocity,
            b.blade_attitude,
            b.airfoil
        )
        if disp:
            print(
                '\r\n  --- Pressure distribution (Body Frame) --- \r\n',
                arrPressure
            )

        F, M = blade.getBladeForceMoment(
            arrPressure,
            b.arrPosWingElems,
            b.chord_length,
            b.blade_length,
            b.n_elements
        )
        if disp:
            print(
                '\r\n  --- F and M of the blade (Body Frame) ---\r\n',
                F, M
            )

        arrFM[id, 0, :] = F
        arrFM[id, 1, :] = M

        sumF += F
        sumM += M

    return sumF, sumM


def func(model, x, F, M):
    """状態量の時間微分の計算."""
    # 加速度の更新
    force_body = F + model.gravityBody()
    accel = model.calcDerivativeOfVelocityBody(force_body)
    # 角加速度の更新
    omega_dot = model.calcDerivativeOfOmegaBody(M)
    # 速度の更新
    vel_inertial = model.getVelocityInertial()
    # クオータニオンの時間微分の更新
    omega_inertial = model.rotationOfPositionVector(x[4:7])
    q_dot = model.calcDerivativeOfQuartanion(omega_inertial)
    f = np.hstack((q_dot, omega_dot, vel_inertial, accel))
    return f


def postProcess(model, x):
    """積分計算後の値をモデルに更新する."""
    model.updateQuartanionODE(x[0:4])
    model.omega_body = x[4:7]
    model.position = x[7:10]
    model.velocityBody = x[10:13]
    print('Quartanion: ', x[0:4])
    print('PQR: ', x[4:7])
    print('Position: ', x[7:10])
    print('Velocity: ', x[10:13])


def main():
    """実行用関数."""
    # クオータニオンモジュールの初期化.
    att = attitude.Attitude6DoF()

    # ブレードの初期化（複数の翼をつける場合は逐次appendする）
    Blades = []
    Blades.append(blade.Blade(
        n_elem=10,
        pos_root=[0.0, 0.06, 0.0],
        att=[0.0 * pi / 180, -80.0 * pi/180, 0.0 * pi/180],
        b_len=0.13,
        c_len=0.1,
        airfoil=af.NACA0012_181127
    ))

    Blades.append(blade.Blade(
        n_elem=10,
        pos_root=[0.0, -0.06, 0.0],
        att=[0.0 * pi / 180, -80.0 * pi/180, 180.0 * pi/180],
        b_len=0.13,
        c_len=0.1,
        airfoil=af.NACA0012_181127
    ))

    # 状態量の初期化
    x0 = np.zeros((13), dtype=float)
    x0[0:4] = att.quartanion
    x0[4:7] = att.omegaBody
    # x0[7:10]: position of the model
    x0[10:13] = att.velocityBody

    t0 = 0.0
    tf = 1.0
    dt = 0.001

    F = np.array([0.0, 0.0, 0])
    M = np.array([0.0, 0.0, 0])

    # ODEの設定
    solver = ode(func).set_integrator('vode', method='bdf')
    solver.set_initial_value(x0, t0)
    solver.set_f_params(att, solver.y, F, M)
    x = np.zeros([int((tf-t0)/dt)+1, 13])
    t = np.zeros([int((tf-t0)/dt)+1, 1])
    F_log = np.zeros((int((tf-t0)/dt)+1, 3))
    M_log = np.zeros((int((tf-t0)/dt)+1, 3))

    index = 0
    while solver.successful() and solver.t < tf:
        solver.integrate(solver.t+dt)
        x[index] = solver.y
        t[index] = solver.t

        print(solver.y)
        F, M = blade_FM(solver.y, Blades)
        F_log[index] = F
        M_log[index] = M

        solver.set_f_params(att, solver.y, F, M)
        postProcess(att, solver.y)
        index += 1

    x_log = np.hstack((t, x))
    F_log = np.hstack((t, F_log))
    M_log = np.hstack((t, M_log))

    np.save('x_1', x_log)
    np.save('M_1', M_log)
    np.save('F_1', F_log)


if __name__ == '__main__':
    main()
