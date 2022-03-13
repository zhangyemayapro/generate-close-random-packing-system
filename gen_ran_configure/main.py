#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:46:47 2022

@author: zhangye
"""
# freud==2.8.0
import hoomd
from hoomd import hpmc
from hoomd import deprecated
import freud
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
# from mpi4py import MPI
# comm=MPI.COMM_WORLD
import sys
sys.path.append("../classes")

import RandomPack
import importlib
# importlib.reload(RandomPack.RandomPack)


np.random.seed(1)
hoomd.context.initialize("")
snapshot = hoomd.data.make_snapshot(N=1000, box=hoomd.data.boxdim(L=10))
box = snapshot.box
frbox = freud.box.Box.from_box(box)
snapshot.particles.position[:] = frbox.wrap(
    np.array([box.Lx, box.Ly, box.Lz])
    * (np.random.random((snapshot.particles.N, 3)) - 0.5)
)
system = hoomd.init.read_snapshot(snapshot)
mc = hpmc.integrate.sphere(seed=42, d=0.000001)
mc.shape_param.set("A", diameter=0.5)
rp = RandomPack.RandomPack(
    system, mc, aimpack=0.6, tol1=0.2, tol2=0.15, maxstep=1000
)



rp.change_nominal_phi(0.4)
rp.get_Totaloverlap2()
rp.routime=int(1e6)
rp.routimeGsd = '../data/routime.gsd'
rp.no_overlapGsd = '../data/no_overlap.gsd'
rp.non_overlap_phi_list.append(rp.phi)

rp.printAcceptanceStatistics()
rp.icmax = 20000


rp.readGsd(rp.no_overlapGsd)
rp.tol1 = 0.002
rp.tol2 = 0.001

while 1:
    # rp.rej_steplast = rp.rej_step
    rp.runMovetrials(step=2000)
    if rp.ic < rp.icmax:
        if rp.maxoverlap < rp.tol1:
            rp.change_nominal_phi(rp.phi+0.001)
        else:
            rp.runMovetrials(step=2000)
    else:
        if rp.maxoverlap < rp.tol1:
            rp.change_nominal_phi(rp.phi+0.001)
        else:
            # while rp.overlap_count != 0:
            while rp.maxoverlap >rp.tol2:
                rp.get_Totaloverlap2()
                rp.change_nominal_phi(rp.phi-0.001)
                rp.runMovetrials(step=500)
                # rp.vibrate(1000)
                rp.get_Totaloverlap2()
            print("dumping")
            rp.dumpPosGsd(rp.no_overlapGsd)
            rp.get_packing_fraction()
            rp.non_overlap_phi_list.append(rp.phi)
            rp.writeLogValue(str(rp.non_overlap_phi_list))
            if (
                abs(rp.non_overlap_phi_list[-1] - rp.non_overlap_phi_list[-2])
                < 0.005
            ):
                rp.tol1 = 0.9*rp.tol1
                rp.tol2 = 0.9*rp.tol2
                if rp.overlap_count == 0:
                    break
            else:
                rp.change_nominal_phi(rp.phi+0.001)
def analize():
    x =[]
    y= []
    for i in range(0,100):
        try:
            rp.readGsd(rp.routimeGsd,frame=-i)
            x.append(rp.phi)
            y.append(rp.maxoverlap)
        except:
            break
    plt.scatter(x,y)
    return x,y
x,y = analize()

rp.get_non_overlap()
