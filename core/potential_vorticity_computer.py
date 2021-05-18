from numba.pycc import CC
from numba.pycc import compiler
from numba import jit
import numpy as np

def compile(verbose=False):
    print("** Compile the pv_computer function with numba")

    cc = CC("compiled_pv")
    cc.verbose = verbose

    @cc.export("pv_computer",
               "void(f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :], f8[:, :, :],f8[:, :, :],f8[:, :, :],f8[:, :, :])")
    def pv_computer(pv,b,w_i,w_j,w_k,av_gradb_i,av_gradb_j,av_gradb_k,av_w_i,av_w_j,av_w_k):

        nz,ny,nx = b.shape
        #computing the gradients.
        #buyoancy is defined at cells centers thus the gradient of b is defined at cells faces
        gradb_k = b[1:nz,:,:]-b[0:nz-1,:,:]
        gradb_j = b[:,1:ny,:]-b[:,0:ny-1,:]
        gradb_i = b[:,:,1:nx]-b[:,:,0:nx-1]

        #defining the gradient of the buyoancy at cells centers by averaging it


        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if i == 0:
                        av_gradb_i[k,j,i] = gradb_i[k,j,i]
                    elif i == nx-1:
                        av_gradb_i[k,j,i] = gradb_i[k,j,i-1]
                    else:
                        av_gradb_i[k,j,i] = (gradb_i[k,j,i]+gradb_i[k,j,i-1])/2

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if j == 0:
                        av_gradb_j[k,j,i] = gradb_j[k,j,i]
                    elif j == ny-1:
                        av_gradb_j[k,j,i] = gradb_j[k,j-1,i]
                    else:
                        av_gradb_j[k,j,i] = (gradb_j[k,j,i]+gradb_j[k,j-1,i])/2

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    if k == 0:
                        av_gradb_k[k,j,i] = gradb_k[k,j,i]
                    elif k == nz-1:
                        av_gradb_k[k,j,i] = gradb_k[k-1,j,i]
                    else:
                        av_gradb_k[k,j,i] = (gradb_k[k,j,i]+gradb_k[k-1,j,i])/2


        #vorticity's components are defined on cells edges

        #av_w_i[k,j,i] = (w_i[k,j,i]+w_i[k,j-1,i]+w_i[k-1,j-1,i]+w_i[k-1,j,i])/4
        #av_w_j[k,j,i] = (w_j[k,j,i]+w_j[k,j,i-1]+w_j[k-1,j,i]+w_j[k-1,j,i-1])/4
        #av_w_k[k,j,i] = (w_k[k,j,i]+w_k[k,j-1,i]+w_k[k,j,i-1]+w_k[k,j-1,i-1])/4

        #defining vorticity's components at cells centers by averaging them
        for k in range(nz):
            if k == 0:
                for j in range(ny):
                    if j == 0:
                        for i in range(nx):
                            if i == 0:
                                av_w_i[k,j,i] = w_i[k,j,i]
                                av_w_j[k,j,i] = w_j[k,j,i]
                                av_w_k[k,j,i] = w_k[k,j,i]
                            else:
                                av_w_i[k,j,i] = w_i[k,j,i]
                                av_w_j[k,j,i] = (w_j[k,j,i]+w_j[k,j,i-1])/2
                                av_w_k[k,j,i] = (w_k[k,j,i]+w_k[k,j,i-1])/2
                    else:
                        for i in range(nx):
                            if i == 0:
                                av_w_i[k,j,i] = (w_i[k,j,i]+w_i[k,j-1,i])/2
                                av_w_j[k,j,i] = w_j[k,j,i]
                                av_w_k[k,j,i] = (w_k[k,j,i]+w_k[k,j-1,i])/2
                            else:
                                av_w_i[k,j,i] = (w_i[k,j,i]+w_i[k,j-1,i])/2
                                av_w_j[k,j,i] = (w_j[k,j,i]+w_j[k,j,i-1])/2
                                av_w_k[k,j,i] = (w_k[k,j,i]+w_k[k,j-1,i]+w_k[k,j,i-1]+w_k[k,j-1,i-1])/4

            else:
                for j in range(ny):
                    if j == 0:
                        for i in range(nx):
                            if i == 0:
                                av_w_i[k,j,i] = (w_i[k,j,i]+w_i[k-1,j,i])/2
                                av_w_j[k,j,i] = (w_j[k,j,i]+w_j[k-1,j,i])/2
                                av_w_k[k,j,i] = w_k[k,j,i]
                            else:
                                av_w_i[k,j,i] = (w_i[k,j,i]+w_i[k-1,j,i])/2
                                av_w_j[k,j,i] = (w_j[k,j,i]+w_j[k,j,i-1]+w_j[k-1,j,i]+w_j[k-1,j,i-1])/4
                                av_w_k[k,j,i] = (w_k[k,j,i]+w_k[k,j,i-1])/2
                    else:
                        for i in range(nx):
                            if i == 0:
                                av_w_i[k,j,i] = (w_i[k,j,i]+w_i[k,j-1,i]+w_i[k-1,j-1,i]+w_i[k-1,j,i])/4
                                av_w_j[k,j,i] = (w_j[k,j,i]+w_j[k-1,j,i])/2
                                av_w_k[k,j,i] = (w_k[k,j,i]+w_k[k,j-1,i])/2
                            else:
                                av_w_i[k,j,i] = (w_i[k,j,i]+w_i[k,j-1,i]+w_i[k-1,j-1,i]+w_i[k-1,j,i])/4
                                av_w_j[k,j,i] = (w_j[k,j,i]+w_j[k,j,i-1]+w_j[k-1,j,i]+w_j[k-1,j,i-1])/4
                                av_w_k[k,j,i] = (w_k[k,j,i]+w_k[k,j-1,i]+w_k[k,j,i-1]+w_k[k,j-1,i-1])/4


        #computing the scalar product giving the potential vorticity pv
        pv[:,:,:] = av_w_i*av_gradb_i+av_w_j*av_gradb_j+av_w_k*av_gradb_k

    cc.compile()

compile()
