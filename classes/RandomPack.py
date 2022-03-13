#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:12:47 2022

@author: zhangye
"""

import numpy as np

# import pandas as pd
from matplotlib import pyplot as plt
import sys
import os

sys.path.append("../")
from time import time
import importlib
import pickle

# importlib.reload("module")

import signac  # 1.2.2
import flow
from flow import FlowProject

import hoomd
import freud

import numpy as np
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import pysnooper


class RandomPack:
    def __init__(self, system, mc, aimpack, tol1, tol2, maxstep):
        """
        Parameters
        ----------
        system : Hoomd System
        mc : Hoomd integrate
        
        maxpack : double
            the needed maximum packing density
        tole : double
            the tolerance overlap ratio
        box : numpy
            np.array([1,1,1])
        maxstep : int
            within maxstep if the maximum step do not drop, decrease the radii
        -------

        """
        self.system = system
        self.mc = mc

        self.tol1 = tol1
        self.tol2 = tol2
        self.maxstep = maxstep
        self.snapshot = self.system.take_snapshot()
        self.snapshot.broadcast()
        self.position = self.snapshot.particles.position
        self.radii = 0.5 * self.snapshot.particles.diameter[0]
        self.sphere_v = 4 * np.pi / 3 * self.radii ** 3
        self.N = self.snapshot.particles.N
        self.hdbox = self.snapshot.box
        self.frbox = freud.box.Box.from_box(system.box)
        self.neighbors = 20

        self.aimpack = aimpack
        self.boxV = self.hdbox.Lx * self.hdbox.Ly * self.hdbox.Lz
        self.aimr = ((self.aimpack * self.boxV / self.N) / (4.0 / 3 * np.pi)) ** (1 / 3)

        self.tot_step = 0
        self.acc_step = 0
        self.rej_step = 0
        self.ic = 0
        self.icmax = int(1e5)
        self.mdisplace = 0.1
        self.set_log()
        # self.get_nlist_forall()
        self.get_Totaloverlap2()
        self.get_packing_fraction()
        self.resultInitialize()
        self.routime = int(1e5)
        self.routimeGsd = "../data/routime.gsd"
        self.changeRflag = False

    def set_log(self, logname="out.log"):
        self.log = open(logname, "w")
        self.writeLogValue(
            "tot_step\t phi\t maxoverlap \t q6lmean\t tol1 \t tol2 \t changeRflag"
        )

    def writeLogValue(self, value):
        self.log.write(str(value) + "\n")
        # if(self.tot_step%10 ==0):
        #     self.log.flush()

    def logging(self):
        self.log.flush()
        self.writeLogValue(
            "%.10d\t" % self.tot_step
            + "%.3f\t" % self.phi
            + "%.5f\t" % self.maxoverlap
            + "%.3f\t" % self.q6lmean
            + "%.5f\t" % self.tol1
            + "%.5f\t" % self.tol2
            + "changeRflag\t"
            + str(self.changeRflag)
        )

    def resultInitialize(self):
        self.maxoverlap_list = []
        self.mean_overlap_list = []
        self.non_overlap_phi_list = []

    def getRfromPhi(self, phi):
        radii = ((phi * self.boxV / self.N) / (4.0 / 3 * np.pi)) ** (1 / 3)
        return radii

    def copyStateToPriorState(self):
        self.radiilast = self.radii
        self.positionlast = self.position
        self.maxoverlaplast = self.maxoverlap

    def copyPriorStateToState(self):
        self.radii = self.radiilast
        self.position = self.positionlast.copy()
        self.maxoverlap = self.maxoverlaplast

    def change_radii(self, change_ratio):
        # self.ic = 0
        self.radii = self.radii * change_ratio
        self.get_Totaloverlap2()
        self.maxoverlaplast = self.maxoverlap
        self.sphere_v = 4 * np.pi / 3 * self.radii ** 3
        print("===change radii===")
        print("the radii is changed by", change_ratio, ",new radii=", self.radii)
        print("maxoverlap=", self.maxoverlap)
        self.changeRflag = True

    def change_nominal_phi(self, newphi, vibrate=True, dump=True):
        self.printAcceptanceStatistics()
        r = self.getRfromPhi(newphi)
        change_ratio = r / self.radii
        self.change_radii(change_ratio=change_ratio)
        if dump:
            self.dumpPosGsd(self.routimeGsd)
        if vibrate:
            self.vibrate(500)

    def get_packing_fraction(self):
        # th definition of packing fraction
        # should we eliminate the overlap ?
        self.phi = (
            4
            / 3
            * np.pi
            * self.radii ** 3
            * self.N
            / (self.hdbox.Lx * self.hdbox.Lz * self.hdbox.Lz)
        )

    def get_nlist2(self, points, query_points, vec=False):
        points = points.reshape((-1, 3))
        query_points = query_points.reshape((-1, 3))

        aq = freud.locality.AABBQuery(self.frbox, points)
        # Here, we ask for the 12 nearest neighbors of each point in query_points.
        #!!!modify
        query_result = aq.query(query_points, dict(r_max=2 * self.radii))
        distances = []
        nlist = []
        for bond in query_result:
            if bond[2] == 0:
                continue
            distances.append(bond[2])
            nlist.append(bond[0:2])
        distances = np.array(distances, dtype=np.float64)
        # distances = np.array(distances).reshape((-1,1))
        nlist = np.array(nlist).reshape((-1, 2))
        if vec:
            if len(nlist) == 0:
                vectors = np.zeros((1, 3))
            else:
                vectors = self.frbox.wrap(
                    query_points[nlist[:, 0]] - points[nlist[:, 1]]
                )
            return nlist, distances, vectors
        return nlist, distances

    def get_len_ratio(self, distances):
        # the overlap regin ia like a optic len
        # if (distances - 2 * self.radii>1e-5).sum() > 0:
        #     raise RuntimeError("(distances > 2 * self.radii).sum()>0")
        len_volume = (
            1
            / 12
            * np.pi
            * (4 * self.radii + distances)
            * (2 * self.radii - distances) ** 2
        )
        len_ratio = len_volume / self.sphere_v
        return len_ratio

    def get_overlap_ratio2(self, points, query_points):
        nlist, distances = self.get_nlist2(points, query_points)
        len_ratio = self.get_len_ratio(distances=distances)
        overlap_ratio = np.zeros((query_points.shape[0], 1))
        for i in range(query_points.shape[0]):
            overlap_ratio[i] = len_ratio[nlist[:, 0] == i].sum()
        return overlap_ratio

    def get_Totaloverlap2(self):
        self.nlist, self.distances = self.get_nlist2(self.position, self.position)
        self.overlap_ratio = self.get_overlap_ratio2(self.position, self.position)
        self.overlap_count = (self.distances < 2 * self.radii).sum() / 2
        self.overlap_par_count = (self.overlap_ratio != 0).sum()
        self.maxoverlap = float(self.overlap_ratio.max())
        self.mean_overlap = self.overlap_ratio.mean()
        # modify
        # self.mdisplace = (1-self.distances[self.distances<2*self.radii]).mean()/2.0
        if  self.distances.shape[0] == 0:
            self.mdisplace = 0
        self.mdisplace = (2 * self.radii - self.distances).mean() / 2.0
        self.get_packing_fraction()
        self.computeOrder()

    # @pysnooper.snoop(2)
    def new_move_reduce_overlap2(self, movepar):
        """
        movepar : int
            self.position[movepar]
        -------
        """
        self.redu_max_flag = False
        self.ic += 1
        self.tot_step += 1
        self.acc_flag = False
        old_pos = self.position[movepar].reshape((1, 3)).copy()
        self.old_pos = old_pos
        loc_nlist, loc_distances, loc_vectors = self.get_nlist2(
            self.position, self.position[movepar], vec=True
        )
        self.loc_nlist, self.loc_distances, self.loc_vectors = (
            loc_nlist,
            loc_distances,
            loc_vectors,
        )

        # if (self.loc_distances == 0).any():
        #     print("error")
        # print(movepar)
        if (loc_distances < 2 * self.radii).sum() == 0:
            # print(movepar)
            # print("no overlap")
            return
        move_vector = loc_vectors[loc_distances < 2 * self.radii].sum(axis=0)
        move_vector = (move_vector / np.linalg.norm(move_vector)).reshape((1, 3))

        # print("move_vec", move_vector)
        # @pysnooper.snoop()
        def optimizeMove(move_ratio):
            # print("move_ratio", move_ratio)
            # in the same time, I shouldn't raise the overlap ratio of others

            new_pos = (old_pos + self.radii * move_ratio * move_vector).reshape((1, 3))
            self.new_pos = new_pos
            self.position[movepar] = self.frbox.wrap(new_pos)
            # print("new_pos", new_pos)
            nloc_nlist, nloc_distances = self.get_nlist2(
                self.position, self.position[movepar].reshape((1, 3))
            )

            self.nloc_nlist, self.nloc_distances = nloc_nlist, nloc_distances
            if nloc_nlist.shape[0] == 0:
                return 0, 0
            overlap_ratio = self.get_overlap_ratio2(
                self.position,
                np.row_stack((self.position[movepar], self.position[nloc_nlist[:, 1]])),
            )
            # print(overlap_ratio)
            # print(self.tot_step,overlap_ratio.max())

            # self.get_Totaloverlap2()
            # return overlap_ratio[0],self.maxoverlap
            return overlap_ratio[0], overlap_ratio.max()

        BINS = 10
        over_new = np.zeros((BINS, 1))
        over_max = np.zeros((BINS, 1))
        move_ratio_a = (np.arange(0, BINS) * self.mdisplace / BINS).reshape((-1, 1))

        for i in range(0, BINS):
            move_ratio = move_ratio_a[i]
            over_new[i], over_max[i] = optimizeMove(move_ratio)

        ok = over_new[over_max < self.maxoverlap]
        if ok.shape[0] == 0:
            self.rej_step += 1
            self.acc_flag = False
            self.position[movepar] = old_pos
            over_new_n, over_max_n = self.vibrateMove(movepar)
            if over_max_n > self.maxoverlap:
                self.position[movepar] = old_pos
            return

        move_ratio = move_ratio_a[np.where(over_new == ok.min())]
        # print(over_max)
        self.acc_step += 1
        self.acc_flag = True

        new_pos = old_pos + self.radii * move_ratio[0] * move_vector
        self.position[movepar] = self.frbox.wrap(new_pos)

        if abs(over_max[0] - self.maxoverlap) < 1e-3:
            # print(abs(over_max[0] - self.maxoverlap))
            self.redu_max_flag = True

        # print('self.tot_step',self.tot_step)
        # print(np.where(over_new == ok.min()),move_ratio)
        # print(old_pos)
        # print(new_pos)
        # nloc_nlist, nloc_distances = self.get_nlist2(
        #     self.position, self.position[movepar].reshape((1, 3))
        # )
        # # self.nloc_nlist, self.nloc_distances = nloc_nlist, nloc_distances
        # if nloc_nlist.shape[0] ==0:
        #     return 0,0
        # overlap_ratio = self.get_overlap_ratio2(
        #     self.position,
        #     np.row_stack(
        #         (self.position[movepar], self.position[nloc_nlist[:, 1]]))
        #     )
        # print("nloc_nlist",loc_nlist)
        # print('overlap_ratio',overlap_ratio)

    # @pysnooper.snoop(depth=2)
    def runMovetrials(self, step):
        step = int(step)
        for i in range(step):
            # self.copyStateToPriorState()
            movepar = np.random.randint(self.N)
            self.movepar = movepar
            self.new_move_reduce_overlap2(movepar)
            # print(self.maxoverlap)
            if self.redu_max_flag:
                self.get_Totaloverlap2()
                self.maxoverlap_list.append((self.tot_step, self.maxoverlap))
                self.mean_overlap_list.append((self.tot_step, self.mean_overlap))

            # if self.maxoverlap - self.maxoverlaplast > 0.0001:
            #     if self.changeRflag:
            #         self.changeRflag = False
            #         break
            #     print(self.tot_step)
            #     print("self.maxoverlap>self.maxoverlaplast")
            #     # self.copyPriorStateToState()
            #     raise RuntimeError("self.maxoverlap>self.maxoverlaplast")
            if self.tot_step % int(self.routime) == 0:
                self.dumpPosGsd(self.routimeGsd)
            if self.tot_step % 100 == 0:
                self.get_Totaloverlap2()
                if self.tot_step % int(1e5) == 0:
                    self.printAcceptanceStatistics()

    # def printAcceptanceStatistics(self):
    #     print("------status------")
    #     print("radii=", self.radii)
    #     print(
    #         "tot_step=",
    #         self.tot_step,
    #         "acc_step=",
    #         self.acc_step,
    #         "rej_step=",
    #         self.rej_step,
    #     )
    #     print("overlap_count=", self.overlap_count)
    #     print("maxoverlap=", self.maxoverlap)
    #     ql6m = self.computeOrder()
    #     print("q6.ql.mean=", ql6m)
    #     # print("self.mean_overlap", self.mean_overlap)
    #     # print("self.mdisplace", self.mdisplace)
    def printAcceptanceStatistics(self):

        print("------status------")
        print("radii=", self.radii)
        print(
            "tot_step=",
            self.tot_step,
            "acc_step=",
            self.acc_step,
            "rej_step=",
            self.rej_step,
        )
        print("overlap_count=", self.overlap_count)
        print("maxoverlap=", self.maxoverlap)

        print("q6.ql.mean=", self.q6lmean)

        # print("self.mean_overlap", self.mean_overlap)
        # print("self.mdisplace", self.mdisplace)
        self.logging()
        self.changeRflag = False

    def vibrateMove(self, movepar, mode="vibrate"):
        old_pos = self.position[movepar].copy().reshape((1, 3))
        if mode == "vibrate":
            new_pos = old_pos + (np.random.random((1, 3)) - 0.5) * self.mdisplace
        # in the same time, I shouldn't raise the overlap ratio of others
        self.new_pos = new_pos
        self.position[movepar] = self.frbox.wrap(new_pos)
        # print("new_pos", new_pos)
        nloc_nlist, nloc_distances = self.get_nlist2(
            self.position, self.position[movepar].reshape((1, 3))
        )
        # self.nloc_nlist, self.nloc_distances = nloc_nlist, nloc_distances
        if nloc_nlist.shape[0] == 0:
            return 0, 0
        overlap_ratio = self.get_overlap_ratio2(
            self.position,
            np.row_stack((self.position[movepar], self.position[nloc_nlist[:, 1]])),
        )
        # print(overlap_ratio)
        # print(self.tot_step,overlap_ratio.max())

        # self.get_Totaloverlap2()
        # return overlap_ratio[0],self.maxoverlap
        return overlap_ratio[0], overlap_ratio.max()

    def vibrate(self, step, movepar=None):
        """
        vibrate to avoid periodic structure
        """
        rej = 0
        for i in range(step):
            # print('yes')
            movepar = np.random.randint(self.N)
            old_pos = self.position[movepar].copy().reshape((1, 3))
            new_overlap, max_overlap = self.vibrateMove(movepar)
            # if max_overlap > self.maxoverlap:
            #     self.position[movepar] = old_pos
            #     rej += 1
        return rej

    def showDiagram(self):
        plt.figure()
        plt.hist(self.overlap_ratio)
        plt.title("r=" + str(self.radii) + " overlap_ratio")
        plt.figure()

        data = np.array(self.maxoverlap_list)
        plt.plot(data[:, 0], data[:, 1])
        plt.title("maxoverlap_list")

        plt.figure()
        data = np.array(self.mean_overlap_list)
        plt.plot(data[:, 0], data[:, 1])

        plt.title("mean_overlap_list")

    def showInjavis(self,filename ):
        from hoomd import hpmc
        from hoomd import deprecated

        hoomd.context.initialize('')
        snapshot = hoomd.data.make_snapshot(N=self.N, box=self.hdbox)
        snapshot.particles.position[:] = self.position
        snapshot.particles.diameter[:] = 2 * self.radii
        system = hoomd.init.read_snapshot(snapshot)
        # mc = hpmc.integrate.sphere(seed=42, d=0.000001)
        # mc.shape_param.set("A", diameter=2 * self.radii)

        deprecated.dump.pos(filename=filename, period=1)
        hoomd.run(1)

    def dumpPosGsd(self, filename):
        # def dumpPosGsd(self,filename = "changeR.gsd"):

        from hoomd import hpmc

        self.gsdfilename = filename
        hoomd.context.initialize("")
        snapshot = hoomd.data.make_snapshot(N=self.N, box=self.hdbox)
        snapshot.particles.position[:] = self.position
        snapshot.particles.diameter[:] = 2 * self.radii
        # print(snapshot.particles.diameter[:])
        system = hoomd.init.read_snapshot(snapshot)
        mc = hpmc.integrate.sphere(seed=42, d=0.000001)
        mc.shape_param.set("A", diameter=2 * self.radii)
        hoomd.dump.gsd(
            filename=filename,
            group=hoomd.group.all(),
            dynamic=["attribute", "property"],
            truncate=False,
            period=1,
        )
        hoomd.option.set_notice_level(0)
        hoomd.run(1)
        # modify
        hoomd.option.set_notice_level(1)

    def readGsd(self, filename="changeR.gsd", frame=-1):
        hoomd.context.initialize("")
        system = hoomd.init.read_gsd(filename, frame=frame)
        self.system = system
        self.snapshot = system.take_snapshot()
        self.snapshot.broadcast()
        self.position = self.snapshot.particles.position
        self.radii = 0.5 * self.snapshot.particles.diameter[0]
        self.sphere_v = 4 * np.pi / 3 * self.radii ** 3
        self.N = self.snapshot.particles.N
        self.hdbox = self.snapshot.box
        self.frbox = freud.box.Box.from_box(system.box)
        self.get_packing_fraction()
        self.get_Totaloverlap2()

    def removeParticle(self, limit):
        print("hhh")
        a = [self.overlap_ratio < limit]
        self.position = self.position[np.column_stack((a, a, a))]
        self.N = self.position.shape[0]
    def get_non_overlap(self):
        while self.overlap_count >0:
            print(self.overlap_count)
            self.change_nominal_phi(self.phi - 0.000001,vibrate=False,dump=False)
    def computeOrder(self, pltChoose=False, meanchoose=False):
        # random ql~0.09
        ql6 = freud.order.Steinhardt(l=6)
        ql6.compute((self.frbox, self.position), {"r_max": 3})
        if pltChoose:
            ql6.plot()
        # return ql6.ql.mean()
        order = ql6.ql.copy()
        order = np.nan_to_num(order, 0)
        self.q6lmean = order[order != 0].mean()
        return order[order != 0].mean()

    # def replaceSnapshotHd(self):

    def computeOrderHd(self, pltChoose=False, meanchoose=False):
        # random ql~0.09
        ql6 = freud.order.Steinhardt(l=6)
        ql6.compute((self.frbox, self.position), {"r_max": 3})
        if pltChoose:
            ql6.plot()
        # return ql6.ql.mean()
        return ql6
