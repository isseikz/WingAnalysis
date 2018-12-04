import numpy as np
import sim
from math import pi
import graph as g

k = 1.0

betas1 = np.array([10.0*pi/180]*10)
sim.betas = betas1
print(sim.betas)

log = sim.sim()
# print(log)
s = log.shape

J1 = log[s[0]-1,1]**2
djdb = 5.0 * pi/180

while True:
    sim.betas = sim.betas - k * djdb
    print(sim.betas)

    log = sim.sim()
    # print(log)
    s = log.shape

    J2 = log[s[0]-1,1]**2
    # print((sim.betas-betas1))

    djdb = (J2-J1)/((sim.betas-betas1))
    print(J2,djdb)

    J1 = J2
    betas1 = sim.betas

    # print(djdb,sim.betas)
    eps = np.dot(djdb,djdb)

    if (eps < 0.00001):
        break
