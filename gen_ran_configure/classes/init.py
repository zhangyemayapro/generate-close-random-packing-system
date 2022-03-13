#!/usr/bin/env python
"""Operations to initialize the data space.

Iterates over all defined state points and initializes
the associated Job."""
import sys
import logging
import argparse
from hashlib import sha1
import random as rand

import signac

sys.path.append("../")

def initialize(job):
    job.init()
    # or:
    # with job:
    #     # initialization routine

rand.seed(1)

def main(args):
    project = signac.get_project()
    statepoints_init = []
    pf_list = []
    for i in range(11):
        pf_list.append(0.4+0.02*i)
    pf_list.sort()
    for pf in pf_list:
        statepoint = {
            'N':            1000,
            'pf':           pf,                   # system packing fraction
            'simulationtype': 'equi',
            'mc_tran':      0.07,
            'mc_rot':       0.04,
            'NPT':          -1,
            'random_seed':  56986,
            'nselect':      1,
            'run_steps':    4e7,
            # 'dump_period':  1e5,
            'restart_period': 1e5,
            'gsd_truncate': False,
            'log_period':   1e4,
            'sdf_xmax':     0.02,
            'sdf_dx':       1e-4,
            'sdf_navg':     2000,
            'sdf_period':   25,
            'msd_period':   1e4
        }
        initialize(project.open_job(statepoint))
        statepoints_init.append(statepoint)
        # Write initialized statepoints additionally to hash table
    project.write_statepoints(statepoints_init)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Initialize the data space.")
    #parser.add_argument(
        #'random',
        #type=str,
        #help="A string to generate a random seed.")
    args = parser.parse_args()

    # Generate an integer from the random str.
    #random_seed = int(sha1(args.random.encode()).hexdigest(), 16) % (10 ** 8)

    logging.basicConfig(level=logging.INFO)
    sys.exit(main(args))
