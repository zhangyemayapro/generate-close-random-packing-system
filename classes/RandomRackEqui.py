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

# import mpi4py
# from mpi4py import MPI
# comm=MPI.COMM_WORLD
import RandomPack

# random_system.RandomPack
class RamdomPackEqui(RandomPack.RandomPack):
    def __init__(self):
        pass

    def initialize(self, filename, frame=-1):
        self.readGsd(filename, frame=frame)
        self.radii = 0.5 * self.snapshot.particles.diameter[0]
        self.sphere_v = 4 * np.pi / 3 * self.radii ** 3
        self.N = self.snapshot.particles.N
        self.hdbox = self.snapshot.box
        self.frbox = freud.box.Box.from_box(self.system.box)
        self.boxV = self.hdbox.Lx * self.hdbox.Ly * self.hdbox.Lz

    def set_log(self):
        self.log = 0.1

    def change_phi(self, newphi):
        hoomd.context.initialize("")
        new_r = self.getRfromPhi(newphi)
        self.radii = new_r
        self.snapshot.particles.diameter[:] = 2 * self.radii
        # print(snapshot.particles.diameter[:])
        self.system = hoomd.init.read_snapshot(self.snapshot)

    def renewPosition():
        return

    def extrapolate(self, s, dx, xmax, degree=5):
        import math

        # determine the number of values to fit
        n_fit = int(math.ceil(xmax / dx))
        s_fit = s[0:n_fit]
        # construct the x coordinates
        x_fit = np.arange(0, xmax, dx)
        x_fit += dx / 2
        # perform the fit and extrapolation
        p = np.polyfit(x_fit, s_fit, degree)
        return (np.polyval(p, 0.0), x_fit, np.polyval(p, x_fit))

    def load_sdf_zy(self, job):
        par = np.loadtxt(job.fn("dump.sdf.dat"))
        # tstep:thetime output sdf
        tstep = par[:, 0]
        # sdf:sdf matrix
        sdf = par[:, 1:]
        #     sdf.shape
        return tstep, sdf

    def p_analyze(self, job):
        """
        structure(str):"FCC"or "BCC"
        """
        P_all = []
        P_ave = []
        i = 0
        d = self.radii * 2
        dx = job.statepoint["sdf_dx"]
        xmax = job.statepoint["sdf_xmax"]
        tstep, sdf = self.load_sdf_zy(job)
        tstep = np.int64(tstep)
        P = np.zeros(sdf.shape[0])
        for j in range(0, sdf.shape[0]):
            phi = job.statepoint["pf"]
            s0, x, sfit = self.extrapolate(sdf[j, :], dx, xmax)
            P[j] = phi * (1 + s0 / (2 * d))
        i += 1
        P_ave.append(P.mean())
        #     print(P.shape)
        #     print("P_ave:",P_ave)
        return tstep, P, P_ave
    

    def calc_pressure(self, time_step):
        dx = self.job.statepoint["sdf_dx"]
        xmax = self.job.statepoint["sdf_xmax"]
        d = 2*self.radii
        r_min = 2*self.radii
        r_max = 2*self.radii*(1+xmax)
        bins = int(xmax/dx)
        rdf = freud.density.RDF(bins=bins, r_max=r_max,r_min=r_min)
        hoomd.util.quiet_status()
        self.snapshot = self.system.take_snapshot()
        self.snapshot.broadcast()
        hoomd.util.unquiet_status()
        rdf.compute(system=self.snapshot, reset=False)
        s = rdf.rdf
        s0, x, sfit = self.extrapolate(s, dx, xmax)
        phi = self.job.statepoint["pf"]
        p = phi * (1 + s0 / (2 * d))
        return p
        #         np.savetxt(
        #     "rdf.csv", np.vstack((rdf.bin_centers, rdf.rdf)).T, delimiter=",", header="r, g(r)"
        # )
        

    def computeOrderq6(self, pltChoose=False, meanchoose=False):
        # random ql~0.09
        hoomd.util.quiet_status()
        self.snapshot = self.system.take_snapshot()
        self.snapshot.broadcast()
        hoomd.util.unquiet_status()
        self.position[:] = self.snapshot.particles.position
        ql6 = freud.order.Steinhardt(l=6)
        ql6.compute((self.frbox, self.position), {"r_max": 3})
        # if pltChoose:
        #     ql6.plot()
        # return ql6.ql.mean()
        order = ql6.ql.copy()
        order = np.nan_to_num(order, 0)
        self.ql6 = ql6
        self.q6lmean = order[order != 0].mean()
        return order[order != 0].mean()

    def computeQ6_a(self,box=None,points=None,l_max = 6):
        if box ==None and points ==None:
            box = self.frbox
            points=self.position
        system = freud.AABBQuery(box, points)  
        num_neighbors = 12
        # nlist = system.query(
        #     points, {"num_neighbors": num_neighbors, "exclude_ii": True}
        # ).toNeighborList()
        nlist = system.query(
            points, {"r_max": self.radii*4, "exclude_ii": True}
        ).toNeighborList()
        
        ld = freud.environment.LocalDescriptors(l_max, mode='global')
        ld.compute(system, neighbors=nlist)
        
        sph = ld.sph.copy()
        self.sph  = sph
        # print(sph,nlist.shape)
        # print('hhh')
        sph = np.nan_to_num(sph,0)
        sph_raw = np.abs(np.mean(sph, axis=0))
        for l in range(l_max,l_max+1):
            Qlm2 = (sph_raw[-(2*l_max+1):])**2
            Ql6 = (4*np.pi/(2*l_max+1)*Qlm2.sum())**0.5
        
        n_b = nlist[:].shape[0]
        # print(Ql6)
        # print(n_b**0.5*Ql6)
        
        return n_b**0.5*Ql6

    def record_step(self,timestep):
        self.job.doc['end_step'] = hoomd.get_step()
        return self.job.doc['end_step'] 
    def runTrails(self, step):
        hoomd.run(step, callback=())


# rpe= RamdomPackEqui()
# rpe.readGsd("../data//no_overlap.gsd",frame=1)
# rpe.radii
