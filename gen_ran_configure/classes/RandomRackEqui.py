#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 04:31:54 2022

@author: zhangye
"""
import hoomd
from hoomd import hpmc
# from hoomd import deprecate
import freud
import numpy as np
import matplotlib.pyplot as plt

# from mpi4py import MPI
# comm=MPI.COMM_WORLD
import RandomPack
# random_system.RandomPack
class RamdomPackEqui(RandomPack.RandomPack):
    def __init__(self):
        self.t=1
    def renewPosition():
        return
    def prepSystem(self):
        return
    def calc_pressure(self):
        return
    def calc_ql6(self):
        return
    def runTrails(self,step):
        hoomd.run(step,callback=())
rpe= RamdomPackEqui()
rpe.readGsd("../data//no_overlap.gsd",frame=1)        
rpe.radii
        
            