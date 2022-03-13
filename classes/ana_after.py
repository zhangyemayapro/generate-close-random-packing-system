#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 23:24:14 2022

@author: zhangye
"""
import os
import signac
import flow

# import init

import sys
from time import time

# for some function was not in this directory
sys.path.append("../")
sys.path.append("../classes")
project = signac.get_project("../")
for job in project:
    sp = job.statepoint

import freud
import hoomd
import matplotlib.pyplot as plt
import numpy as np
from hoomd import hpmc
import RandomRackEqui


hoomd.context.initialize("")
# hoomd.option.set_notice_level(1)
rpe = RandomRackEqui.RamdomPackEqui()
rpe.project = signac.get_project("")
for job in rpe.project.find_jobs({'pf':0.58}):
    rpe.job = job
    sp = job.statepoint
    print(sp)
    with job:
        if job.isfile(job.fn("restart.gsd")):
            continue
        if job.isfile(job.fn("restart.gsd")):
            rpe.initialize(job.fn("restart.gsd"))
        else:
            rpe.initialize(rpe.project.fn("../data/true_non_overlap.gsd"))
        rpe.change_phi(sp["pf"])
        mc = hpmc.integrate.sphere(seed=42, d=sp["mc_tran"])
        mc.shape_param.set("A", diameter=2 * rpe.radii)
        # Equilibrate the system a bit before accumulating the RDF.
        # hoomd.run(1e4)
        logger = hoomd.analyze.log(
            filename="output.log",
            quantities=["q6",'p','hpmc_translate_acceptance','end_step'],
            period=sp['log_period'],
            header_prefix="#",
            overwrite=True,
        )
        logger.register_callback("q6", rpe.computeOrderq6)
        logger.register_callback("p", rpe.calc_pressure)
        logger.register_callback("end_step", rpe.record_step)
        hpmc.analyze.sdf(
            mc=mc, filename="dump.sdf.dat", xmax=sp['sdf_xmax'], 
            dx=sp['sdf_dx'], navg=sp['sdf_navg'], period=sp['sdf_period']
        )

        gsd_restart = hoomd.dump.gsd(
            filename="restart.gsd",
            group=hoomd.group.all(),
            truncate=sp["gsd_truncate"],
            period=int(sp["restart_period"]),
        )
        # Set up SDF logging
        hpmc.analyze.sdf(
            mc=mc,
            filename="dump.sdf.dat",
            xmax=sp["sdf_xmax"],
            dx=sp["sdf_dx"],
            navg=sp["sdf_navg"],
            period=sp["sdf_period"],
        )
        # dump the shape information into the trajectory log
        hoomd.option.set_notice_level(1)
        run_steps = int(sp["run_steps"])
        hoomd.run(run_steps)
        
        # Store the computed RDF in a file
        tstep, p, p_ave = rpe.p_analyze(job)
        t_p=np.column_stack((tstep,p))
        np.savetxt(job.fn('p.txt'),t_p)
        
