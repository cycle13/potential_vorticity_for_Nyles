from numba.pycc import CC
from numba.pycc import compiler
from numba import jit
import numpy as np

def compile(verbose=False):
    print("** Compile the pv_computer function with numba")

    cc = CC("compiled_pv")
    cc.verbose = verbose

    @cc.export("pv_computer",
               "void(float64, f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :])")
    def pv_computer(f, pv, b, w_i, w_j, w_k):
        """ Compute the pv

        pv = del b . omega

        where b is the buoyancy and omega the vorticity (vector)

        omega = (w_i, w_j, w_k+f)

        f, the Coriolis parameter, is added to the vertical component

        Parameters
        ----------
        f: float
        pv, b, w_i, w_j, w_k: arrays
        """

        nz,ny,nx = b.shape

        # loop only on points where we can compute both del b and the
        # vorticity average -> exclude first and last elements of each
        # axis
        for k in range(1,nz-1):
            for j in range(1,ny-1):
                for i in range(1,nx-1):
                    # computing the gradients.  buyoancy is defined at
                    # cells centers thus the gradient of b is defined
                    # at cells faces defining the gradient of the
                    # buyoancy at cells centers by averaging it

                    av_gradb_i = b[k,j,i+1]-b[k,j,i-1] # *0.5 added later
                    av_w_i = (w_i[k,j,i]+w_i[k-1,j,i]+w_i[k,j-1,i]+w_i[k-1,j-1,i]) # 0.25 added later

                    av_gradb_j = b[k,j+1,i]-b[k,j-1,i]
                    av_w_j = (w_j[k,j,i]+w_j[k-1,j,i]+w_j[k,j,i-1]+w_j[k-1,j,i-1])

                    av_gradb_k = b[k+1,j,i]-b[k-1,j,i]
                    av_w_k = (w_k[k,j,i]+w_k[k,j-1,i]+w_j[k,j,i-1]+w_j[k,j-1,i-1]+4*f)

                    pv[k,j,i] = (av_w_i*av_gradb_i+av_w_j*av_gradb_j+av_w_k*av_gradb_k) * 0.125 # 0.5*0.25
    cc.compile()

compile()
