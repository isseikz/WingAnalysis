import numpy as np
from scipy.integrate import ode
from math import atan, pi, sqrt, cos, sin, atan2

def x2(x):
    return -(x-1)**2 + 1

def test():
    rho = 1.293
    dxdt = 2.0
    v = pi * 1.0
    c = 0.1
    dz = 0.01
    Cl = 1.1
    Cd = 0.1
    beta = 80 *pi/180
    z = 1.0
    omega = pi

    # dFvNdFh(rho, z, omega, dxdt, c, dz, Cl, Cd, beta)

    zn = 1
    z0 = 0.5
    dz = 0.01
    FvNFh(zn, z0, dz, rho, omega, dxdt, c, Cl, Cd, beta)

tf = 5.0
dt = 0.001
rho = 1.293

# 翼の条件
beta = 70 *pi/180

zn = 0.19
z0 = 0.06
c = 0.1
dz = 0.01

# 機体の条件
Cdb = 1.0
m = 0.094   # [kg]
I = 0.00020633122  # [kg m^2] (11/30/2/3.14)^2*0.19^2*0.094*9.81/0.55
g = 9.81
nw = 2

log = np.zeros((int(tf/dt), 9)) # [t, x, dxdt, w, psi, acc, Fv, M]
logDF = np.zeros((int(tf/dt)+1,int((zn - z0)/dz)),dtype=float)
logDM = np.zeros((int(tf/dt)+1,int((zn - z0)/dz)),dtype=float)
id = int(0)

def sim():
    global tf,dt,rho,beta,zn,z0,c,dz,Cdb,m,I,g,nw, id,logDF,log

    np.save('settings.npy',np.array([tf,dt,rho,beta,zn,z0,c,dz,Cdb,m,I,g,nw]))

    x = np.array([0.0,0.0,0.0,0.0]) # [dxdt, x, w, psi]
    solver = ode(func).set_integrator('dopri5')
    solver.set_initial_value(x)



    while solver.successful() & (solver.t < tf):
        omega = x[2]
        dxdt  = x[0]

        FM = FvNM2(zn, z0, dz, rho, omega, dxdt, c, beta)#FvNM2(zn, z0, dz, rho, omega, dxdt, c, beta)
        Fv = FM[0]
        M  = FM[1]
        Fd = - 0.5 * rho * x[0] ** 2 * ((0.04 ** 2) * pi) * Cdb

        d2xdt2 =  Fv * nw / m + g + Fd / m
        dwdt   = (M * nw ) / I

        if id < int(tf/dt):
            log[id, 0]   = solver.t
            log[id, 1:5] = x
            log[id, 5]   = d2xdt2
            log[id, 6]   = Fv
            log[id, 7]   = M
            log[id, 8]   = Fd
            id += 1

        solver.set_f_params(x, d2xdt2, dwdt)
        solver.integrate(solver.t+dt)

        sol = solver.y
        # print(solver.t,sol)
        x = sol

    # print(f'finished! mv^2/2={m*x[0]**2/2}, Iw^2/2={I*x[2]**2/2}')
    # print(f'x={x}')

    np.save('out.npy', log)
    np.save('df.npy', logDF)
    np.save('dm.npy', logDM)
    return x

def func(x, d2xdt2, dwdt):
    return np.array([d2xdt2, x[0], dwdt, x[2]])

def NACA0012(alpha):
    alphaDeg = alpha * 180 / pi
    Cd = 0.1
    threshold = 12
    if (alphaDeg >= threshold) | (alphaDeg <= -threshold):
        Cl = 0.5 * np.sign(alpha)
    elif ( -threshold < alphaDeg) & (alphaDeg < threshold):
        Cl = alphaDeg / 10
    else:
        print('Error')
        Cl = alphaDeg / 10
    # print(f'Alpha: {alpha}, Cl: {Cl}')
    return Cl, Cd

def NACA0012_181127(alpha):
    alphaDeg = alpha * 180 / pi

    threshold = 12
    if (alphaDeg >= threshold) | (alphaDeg <= -threshold):
        Cl = 0.5 * np.sign(alpha)
    elif ( -threshold < alphaDeg) & (alphaDeg < threshold):
        Cl = alphaDeg / 10
    else:
        print('Error')
        Cl = alphaDeg / 10

    if (alphaDeg >= threshold) | (alphaDeg <= -threshold):
        Cd = 2.0 * sin(alpha)
    elif ( -threshold < alphaDeg) & (alphaDeg < threshold):
        Cd = 0.1
    else:
        print('Error')
        Cd = alphaDeg / 10
    # print(f'Alpha: {alpha}, Cl: {Cl}')
    return Cl, Cd

def FvNM(zn, z0, dz, rho, omega, dxdt, c, beta):
    global betaLog
    sum = 0.0
    ne = int((zn - z0)/dz/2)
    for n in range(0,ne):
        k = 2*n
        z1 = z0 + dz *  k
        z2 = z0 + dz * (k+1)
        z3 = z0 + dz * (k+2)
        beta1 = beta + (z1-z0)/(zn-z0)* -0*pi/180
        beta2 = beta + (z2-z0)/(zn-z0)* -0*pi/180
        beta3 = beta + (z3-z0)/(zn-z0)* -0*pi/180
        dSum = (dFvNdM(rho, z1, omega, dxdt, c, dz, beta1) + 4*dFvNdM(rho, z2, omega, dxdt, c, dz, beta2) + dFvNdM(rho, z3, omega, dxdt, c, dz, beta3))* dz / 3
        logDF[id,n] = dSum[0]
        sum += dSum
    res = sum
    logDF[id,ne] = sum[0]
    print(f'F,M = {res}')
    return np.array(res)

def betaToConstantAlpha(angleDeg, dxdt, v):
    alpha = angleDeg * pi / 180
    beta = atan2(dxdt, v) - alpha
    return beta

def betaFromSteadyPoint(phi, dxdt, v,z):
    global z0,zn
    beta = 0.0
    section = (zn-z0)/6
    dist = (z-z0)
    if dist < 1 * section:
        beta = 2.91376693
    elif 1 * section <= dist and dist < 2 * section:
        beta = -0.24244264
    elif 2 * section <= dist and dist < 3 * section:
        beta = -2.16667532
    elif 3 * section <= dist and dist < 4 * section:
        beta = -3.45976006
    elif 4 * section <= dist and dist < 5 * section:
        beta = -4.38759973
    elif 5 * section <= dist:
        beta = -5.08545683
    beta *= pi/180
    return beta

def betaFromInput(z):
    global z0,zn,betas
    # print(betas)
    section = (zn-z0)/len(betas)
    dist = (z-z0)

    beta = betas[int(dist/section)]
    # print(beta)
    return beta

def alphaFromPhiBeta(phi, beta):
    alpha = phi - beta
    return alpha

def alphaBetaFixedWing(betaDeg):
    beta = betaDeg * pi / 180
    return beta

def dynamicPressure(rho, ux, uy):
    p = 0.5 * rho * (ux**2 + uy**2)
    return p

def aerodynamicForce(p, c, Cl, Cd):
    # print(p, c, Cl, Cd)
    dL = - p * c * Cl
    dD = p * c * Cd
    dF = sqrt(dL**2 + dD**2)
    return dL, dD, dF

def FvMFrom(phi, dL, dD, z):
    cosb = cos(phi)
    sinb = sin(phi)
    rot = np.array([
    [cosb, -sinb],
    [sinb,  cosb]
    ])
    vec = np.dot(rot, np.array([-dD,dL]))

    dM  = vec[0] * z
    dFv = vec[1]
    return dFv, dM

def FvNM2(zn, z0, dz, rho, omega, dxdt, c, beta):
    global betaLog
    sum = 0.0
    ne = int((zn - z0)/dz/2)

    arrWingZ                     = np.arange(z0,zn,dz,dtype=float)
    arrWingV                     = arrWingZ * omega
    arrWingPhi                   = np.frompyfunc(atan2, 2, 1)(dxdt, arrWingV)
    # arrWingBeta                  = np.frompyfunc(betaFromSteadyPoint, 4, 1)(arrWingPhi, dxdt, arrWingV, arrWingZ)
    arrWingBeta                  = np.frompyfunc(betaToConstantAlpha, 3, 1)(12, dxdt, arrWingV)
    # arrWingBeta                  = np.frompyfunc(betaFromInput, 1, 1)(arrWingZ)
    # arrWingBeta                  = np.frompyfunc(alphaBetaFixedWing, 1, 1)(80)#0.20942928*180/pi
    arrWingAlpha                 = np.frompyfunc(alphaFromPhiBeta, 2,1)(arrWingPhi,arrWingBeta)
    # print(arrWingPhi,arrWingBeta,arrWingAlpha)
    arrWingCl, arrWingCd         = np.frompyfunc(NACA0012_181127,1,2)(arrWingAlpha)
    arrWingP                     = np.frompyfunc(dynamicPressure, 3,1)(rho, dxdt, arrWingV)
    arrWingL, arrWingD, arrWingF = np.frompyfunc(aerodynamicForce, 4,3)(arrWingP,c,arrWingCl, arrWingCd)
    arrWingFv, arrWingM          = np.frompyfunc(FvMFrom, 4, 2)(arrWingPhi, arrWingL, arrWingD, arrWingZ)
    # print(arrWingBeta)

    Fv = 0
    M  = 0
    for n in range(0,ne):
        k = 2*n
        Fv += arrWingFv[k] + 4 * arrWingFv[k+1] + arrWingFv[k+2]
        M  += arrWingM[k]  + 4 * arrWingM[k+1]  + arrWingM[k+2]
    FvM = np.array([Fv,M]) * (dz/3)
    res = FvM

    logDF[id] = arrWingFv
    logDM[id] = arrWingM
    # print(logDF[id])
    # betaLog[id,ne]   = FvM

    return np.array(res)

def dFvNdM(rho, z, omega, dxdt, c, dz, beta):
    v = z * omega
    phi = atan2(dxdt, v)
#
    beta = alphaBetaFixedWing(10, dxdt, v)
    # beta = betaToConstantAlpha(20, dxdt, v)
    # beta = betaFromSteadyPoint(phi, dxdt, v, z)
    alpha = alphaFromPhiBeta(phi, beta)
    # print(f'beta: {beta*180/pi}')
    # print(alpha,v,dxdt)

    Cl, Cd = NACA0012(alpha)

    p = 0.5 * rho * (v**2 + dxdt**2)
    dL = - p * c * Cl
    dD = p * c * Cd
    dF = sqrt(dL**2 + dD**2)

    cosb = cos(phi)
    sinb = sin(phi)
    rot = np.array([
    [cosb, -sinb],
    [sinb,  cosb]
    ])
    vec = np.dot(rot, np.array([-dD,dL]))

    dM  = vec[0] * z
    dFv = vec[1]

    # print(f'alpha={alpha*180/pi}, beta={beta*180/pi} dFv={dFv}, dFh={dM}')
    return np.array([dFv, dM])

betas = np.array([10.0*pi/180]*10)
def betaOptimization():
    global betas, log

    k = 0.0001

    betas1 = np.array([10.0*pi/180,10.0*pi/180,10.0*pi/180,10.0*pi/180,10.0*pi/180,10.0*pi/180,10.0*pi/180,10.0*pi/180,10.0*pi/180,10.0*pi/180])
    betas = betas1

    J1 = sim()[1]**2
    # print(J1)
    djdb = np.array([1.0*pi/180,0.9*pi/180,0.8*pi/180,0.7*pi/180,0.6*pi/180,0.5*pi/180,0.4*pi/180,0.3*pi/180,0.2*pi/180,0.1*pi/180])

    while True:
        betas2 = betas - np.dot(k,djdb)
        betas = betas2

        J2 = sim()[1]**2
        djdb = (J2-J1)/(betas2-betas1)
        # print((betas-betas1))
        print('opt:',J2,betas)

        J1 = J2
        betas1 = betas

        # print(djdb,sim.betas)
        eps = np.dot(djdb,djdb.T)

        if (eps < 0.00001):
            break

    # print(betas)
    np.save('betas.npy',betas)




if __name__ == '__main__':
    # betas = np.array([0.20942928, 0.20646174, 0.20356289, 0.20076218, 0.19810868, 0.19569071, 0.19368492, 0.19250348, 0.19338294, 0.20250597])
    # betas = np.array([0.20942928, 0.20942928, 0.20942928, 0.20942928, 0.20942928, 0.20942928, 0.20942928, 0.20942928, 0.20942928, 0.20942928])
    betas = np.array([-0.2244577,	-0.19882606,	-0.1750717,	-0.15399918,	-0.13694,	-0.126336,	-0.12698655,	-0.15016448,	-0.22966114,	-0.5344328])
    sim()
    # betaOptimization()
