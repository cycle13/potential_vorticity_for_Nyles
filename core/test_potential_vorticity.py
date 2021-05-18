import numpy as np
import variables as var
from nyles import Nyles
import model_les
import parameters as p


param = p.UserParameters()
param.discretization["global_nx"] = 12
param.discretization["global_ny"] = 24
param.discretization["global_nz"] = 32

nx = param.discretization["global_nx"]
ny = param.discretization["global_ny"]
nz = param.discretization["global_nz"]

nyles = Nyles(param)
state = nyles.model.state


b = state.b.view('i')
w_i = state.vor["i"].view('i')
w_j = state.vor["j"].view('i')
w_k = state.vor["k"].view('i')
print("nz,ny,nx: ",nz,ny,nx)
print("b: ",b.shape)
print("w_i",w_i.shape)
print("w_j",w_j.shape)
print("w_k",w_k.shape)

#adding the coriolis force
rotating = param.physics["rotating"]
if rotating == True:
    f = param.physics["coriolis"]
else:
    f = 0

w_k[:, :-1, :-1] += f

#computing the gradients.
#buyoancy is defined at cells centers thus the gradient of b is defined at cells faces
gradb_k = b[1:,:,:]-b[0:nz-1,:,:]
gradb_j = b[:,1:ny,:]-b[:,0:ny-1,:]
gradb_i = b[:,:,1:nx]-b[:,:,0:nx-1]

print("gradb_k: ",gradb_k.shape)
print("gradb_j: ",gradb_j.shape)
print("gradb_i: ",gradb_i.shape)

#defining the gradient of the buyoancy at cells centers by averaging it
av_gradb_i = np.full((nz,ny,nx),0)
av_gradb_j = np.full((nz,ny,nx),0)
av_gradb_k = np.full((nz,ny,nx),0)

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
                av_gradb_k[k,j,i] = w_k[k,j,i]*gradb_k[k,j,i]
            elif k == nz-1:
                av_gradb_k[k,j,i] = w_k[k,j,i]*gradb_k[k-1,j,i]
            else:
                av_gradb_k[k,j,i] = w_k[k,j,i]*(gradb_k[k,j,i]+gradb_k[k-1,j,i])/2




pv = np.full((nz,ny,nx),0)


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
