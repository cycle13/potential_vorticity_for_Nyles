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
f=Dataset(ncfile)

def read_param(ncfile):
    """ Retrieve all the parameters stored in the history file

    Parameters
    ----------
    ncfile : str
          the name of the NetCDF file

    Returns
    -------
    param : dict
         a dictionary of the experiment parameters
    """
    integers = "0123456789"
    param = {}
    with Dataset(ncfile, "r") as nc:
        param_list = nc.ncattrs()
        # print(param_list)
        for p in param_list:
            val = nc.getncattr(p)
            if type(val) is str:
                if val in ["False", "True"]:
                    val = (val == "True")
                elif "class 'list" in val:
                    val = val.split('>:')[-1].strip()
                    val = val[1:-1].split(', ')
                    if val[0][0] in integers:
                        val = [int(e) for e in val if e[0] in integers]
                    elif val[0][0] is "'":
                        val = [e.strip("'") for e in val]
            param[p] = val
    return param

param = read_param(ncfile)
dx = param["Lx"]/param["global_nx"]
dy = param["Ly"]/param["global_ny"]
dz = param["Lz"]/param["global_nz"]

nh = param["nh"]
nz, ny, nx = param["global_nz"], param["global_ny"], param["global_nx"]

if "x" in param["geometry"]:
    dom_x = slice(nh, nh+nx)
else:
    dom_x = slice(nx)

if "y" in param["geometry"]:
    dom_y = slice(nh, nh+ny)
else:
    dom_y = slice(ny)

if "z" in param["geometry"]:
    dom_z = slice(nh, nh+nz)
else:
    dom_z = slice(nz)

with Dataset(ncfile) as nc:
    # remember: (u,v,w) in the model are the *covariant*
    # components of the velocity. Their dimension is L^2 T^-1
    # the "real" velocity components are obtained by division
    # with the cell lengths dx, dy and dz
    u4 = nc.variables["u"][:, dom_z, dom_y, dom_x]/dx
    v4 = nc.variables["v"][:, dom_z, dom_y, dom_x]/dy
    w4 = nc.variables["w"][:, dom_z, dom_y, dom_x]/dz
    b4 = nc.variables["b"][:, dom_z, dom_y, dom_x]
    time = nc.variables["t"][...]
    x = nc.variables["x"][dom_x]
    y = nc.variables["y"][dom_y]
    z = nc.variables["z"][dom_z]

nt, nz, ny, nx = np.shape(u4)
print("the shape of 4D variables is: ", u4.shape)
# as you can see nx, ny, nz are not necessarily equal to
print("nx=%i / nx_glo=%i" % (nx, param["global_nx"]))
print("ny=%i / ny_glo=%i" % (ny, param["global_ny"]))
print("nz=%i / nz_glo=%i" % (nz, param["global_nz"]))

# this is because of the "halo".
# If a dimension is periodic then an extra halo is added on the left
# and the right. The halo width is in
nh = param["nh"]

# Watch out, when you do averages over the domain, you don't want to
# double count these points ...


def get_slice(n, nglo, nh, direction):
    """ return the slice that spans the interior elements,
    in one direction, detecting whether there is a halo or not
    """
    if n == nglo:
        idx = slice(0, n)
        print("closed in %s" % direction)
    else:
        idx = slice(nh, n+nh)
        print("periodic in %s" % direction)
    return idx


xidx = get_slice(nx, param["global_nx"], nh, "x")
yidx = get_slice(ny, param["global_ny"], nh, "y")

print(np.shape(u4))











#general information about the ncfile
print(f,'\n')
#print(f.variables.keys(),'\n')



#retrieving data from ncfile
t=f.variables['t']
t=t[:]

b=f.variables['b']
b=b[:]

u=f.variables['u']
u=u[:]

v=f.variables['v']
v=v[:]

w=f.variables['w']
w=w[:]

vor_i=f.variables['vor_i']
vor_i=vor_i[:]

vor_j=f.variables['vor_j']
vor_j=vor_j[:]

vor_k=f.variables['vor_k']
vor_k=vor_k[:]

nt, nz, ny, nx = np.shape(u)
print("the shape of 4D variables is: ", u.shape)

pv = f.variables['pv']
pv = pv[:]

#diff = (pv[1:,:,:,:]-pv[:nt-1,:,:,:])
#print(diff)

print(pv)
