# -*- coding: utf-8 -*-
"""翼型の空力係数の近似値モジュール."""

import numpy as np
from math import pi, sin


def NACA0012(alpha):
    """NACA0012."""
    alphaDeg = alpha * 180 / pi
    Cd = 0.1
    threshold = 12
    if (alphaDeg >= threshold) | (alphaDeg <= -threshold):
        Cl = 0.5 * np.sign(alpha)
    elif (-threshold < alphaDeg) & (alphaDeg < threshold):
        Cl = alphaDeg / 10
    else:
        print('Error')
        Cl = alphaDeg / 10
    # print(f'Alpha: {alpha}, Cl: {Cl}')
    return Cl, Cd


def NACA0012_181127(alpha):
    """NACA0012改良版."""
    alphaDeg = alpha * 180 / pi

    threshold = 12
    if (alphaDeg >= threshold) | (alphaDeg <= -threshold):
        Cl = 0.5 * np.sign(alpha)
    elif (-threshold < alphaDeg) & (alphaDeg < threshold):
        Cl = alphaDeg / 10
    else:
        print('Error')
        Cl = alphaDeg / 10

    if (alphaDeg >= threshold) | (alphaDeg <= -threshold):
        Cd = 2.0 * sin(alpha)
    elif (-threshold < alphaDeg) & (alphaDeg < threshold):
        Cd = 0.1
    else:
        print('Error')
        Cd = alphaDeg / 10
    # print(f'Alpha: {alpha}, Cl: {Cl}, Cd: {Cd}')
    return Cl, Cd
