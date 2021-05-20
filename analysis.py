import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.interpolate import griddata



datadir = "/users/bertrandducrocq/data/Nyles"
expname = "ekman_2"
ncfile = "%s/%s/%s_00_hist.nc" % (datadir, expname, expname)
#ncfile = "%s/%s/potential_vorticity.nc" % (datadir, expname)


f=Dataset(ncfile)

pv = f.variables["pv"]
pv = pv[:]


nt,nz,ny,nx = pv.shape
print(pv.shape)

"""
t=0
z=int(nz/2)


#creation & configuration of the figure
fig,ax=plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(0,nx)
ax.set_ylim(0,nz)



ax.contourf(np.arange(0,nx,1), np.arange(0,ny,1), pv[t,z,:,:],cmap='jet')
fig.colorbar(ax.contourf(np.arange(0,nx,1), np.arange(0,ny,1), pv[t,z,:,:],cmap='jet'))



def animate(i):
    ax.cla()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax.contourf(np.arange(0,nx,1), np.arange(0,ny,1), pv[i,z,:,:],cmap='jet')


anim=animation.FuncAnimation(fig,animate,frames=nt,interval=100,repeat=False,blit=False)

plt.show()
"""
