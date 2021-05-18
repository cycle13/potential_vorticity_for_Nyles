import numpy as np
import sys
from mpi4py import MPI

def get_myrank(procs=None):
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()

    if procs is None:
        pass
    else:
        msg = 'use mpirun -np %i python ' % np.prod(procs) + ' '.join(sys.argv)
        assert comm.Get_size() == np.prod(procs), msg

    return myrank

def abort():
    comm = MPI.COMM_WORLD
    comm.Abort()

def barrier():
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    comm.barrier()
    if myrank == 0:
        print("", flush=True, end="")

def global_sum(localsum):
    return MPI.COMM_WORLD.allreduce(localsum, op=MPI.SUM)

def global_max(localmax):
    return MPI.COMM_WORLD.allreduce(localmax, op=MPI.MAX)
