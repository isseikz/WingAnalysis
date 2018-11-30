import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from math import pi

import sim
import pandas as pd

def loadCSV(filepath):
    data = pd.read_csv(filepath)
    # print(data)
    return np.array(data)

def loadSim():
    data = np.load('out.npy')
    dataDF = np.load('df.npy')
    dataDM = np.load('dm.npy')
    # print(data)
    return data, dataDF, dataDM

if __name__ == '__main__':
    dataExp = loadCSV('181113_wingfall_without_tether_1.csv')
    dataSim, dataDF, dataDM = loadSim()
    # print(dataDF)

    fig, ax = plt.subplots()
    ax.plot(dataSim[:,0],dataSim[:,1:8])
    plt.xlabel("time (s)", fontsize=10)
    ax.legend(['dxdt [m/s]','x [m]','w [rad/s]','angle [rad]', 'acc [m/s^2]','Fv[N]','M[Nm]'],fontsize=12,loc='upper left')
    ax.grid()
    # plt.show()

    fig0, ax0 = plt.subplots()
    ax0.plot(dataSim[:,0],dataSim[:,5]/9.81-1,'b--', dataSim[:,0],dataSim[:,3]/(2*pi),'r--',dataExp[48*300+200:49*300+200,0]-dataExp[48*300+200,0],dataExp[48*300+205:49*300+205,1],'b',dataExp[48*300+205:49*300+205,0]-dataExp[48*300+205,0],dataExp[48*300+205:49*300+205,4]/(360),'r')
    ax0.set(xlim=[0,0.5],ylim=[-0.2,2])
    plt.xlabel("time (s)", fontsize=15)
    plt.tick_params(labelsize=18)
    ax0.legend(['acc(Sim) [G]','w(Sim) [RPS]','acc(Exp) [G]','w(Exp) [RPS]'],fontsize=17,loc='upper left')
    ax0.grid()
    plt.rcParams["font.size"] = 24
    fig0.tight_layout()
    # plt.show()

    fig1, ax1 = plt.subplots()
    ax1.plot(dataSim[:,0],dataSim[:,5]/9.81-1,'b--', dataSim[:,0],dataSim[:,3]/(2*pi),'r--',dataExp[48*300+200:49*300+200,0]-dataExp[48*300+200,0],dataExp[48*300+205:49*300+205,1],'b',dataExp[48*300+205:49*300+205,0]-dataExp[48*300+205,0],dataExp[48*300+205:49*300+205,4]/(360),'r')
    ax1.set(xlim=[0,0.5],ylim=[-0.2,2])
    plt.xlabel("time (s)", fontsize=10)
    plt.tick_params(labelsize=12)
    ax1.legend(['acc(Sim) [G]','w(Sim) [RPS]','acc(Exp) [G]','w(Exp) [RPS]'],fontsize=12,loc='upper left')
    ax1.grid()
    plt.rcParams["font.size"] = 12
    fig1.tight_layout()
    # plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(dataSim[:,0],dataSim[:,1])
    ax2.plot(dataSim[:,0],dataSim[:,3])
    plt.xlabel("time (s)", fontsize=10)
    ax2.legend(['dxdt [m/s]','w [rad/s]'],fontsize=12,loc='upper left')
    ax2.grid()
    # plt.show()

    fig3, ax3 = plt.subplots()
    ax3.plot(dataSim[:,0],dataSim[:,5])
    ax3.plot(dataSim[:,0],dataSim[:,6]/0.094)
    ax3.plot(dataSim[:,0],dataSim[:,8]/0.094)
    plt.xlabel("time (s)", fontsize=15)
    plt.ylabel("acceleration", fontsize=15)
    plt.tick_params(labelsize=18)
    ax3.legend(['acc [m/s^2]','F_vertival[m/s^2]','F_v_body [m/s^2]'],fontsize=17,loc='upper right')
    # ax3.set(xlim=[0,5.0],ylim=[-6,10])
    ax3.grid()

    fig4, ax4 = plt.subplots()
    x = range(dataDF.shape[1]-1)
    y = np.arange(start=0,stop=5+1/dataDF.shape[0], step=5/dataDF.shape[0], dtype=float)#range(dataDF.shape[0])
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, dataDF[:,0:dataDF.shape[1]-1])
    # plt.axis("image")
    plt.colorbar()

    fig5, ax5 = plt.subplots()
    x = range(dataDM.shape[1]-1)
    y = np.arange(start=0,stop=5+1/dataDM.shape[0], step=5/dataDM.shape[0], dtype=float)#range(dataDF.shape[0])
    X, Y = np.meshgrid(x, y)
    plt.pcolormesh(X, Y, dataDM[:,0:dataDF.shape[1]-1])
    # plt.axis("image")
    plt.colorbar()

    plt.show()
