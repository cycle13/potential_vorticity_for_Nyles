import numpy as np
from netCDF4 import Dataset
import compiled_pv as cpv


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

#retrieving data from ncfile
t=f.variables['t']
t=t[:]

b=f.variables['b']
b=b[:]

w_i=f.variables['vor_i']
w_i=w_i[:]

w_j=f.variables['vor_j']
w_j=w_j[:]

w_k=f.variables['vor_k']
w_k=w_k[:]

nt,nz,ny,nx =  b.shape

#pv = np.full((nz-1,ny-1,nx-1),0)
pv = np.full((nz,ny,nx),0)


#adding the coriolis force
rotating = param["rotating"]
if rotating == True:
    f = param["coriolis"]
else:
    f = 0.

fn = "%s/%s/potential_vorticity.nc" % (datadir, expname)
ds = Dataset(fn, 'w', format='NETCDF4')

time = ds.createDimension('time', None)
x_dim = ds.createDimension('x', nx)
y_dim = ds.createDimension('y', ny)
z_dim = ds.createDimension('z', nz)

stored_pv = ds.createVariable('potential_vorticity',np.float64, ('time','z','y','x'))

for t in range(nt):
    cpv.pv_computer(f,pv,b[t,:,:,:],w_i[t,:,:,:],w_j[t,:,:,:],w_k[t,:,:,:])
    stored_pv[t,:,:,:] = pv
