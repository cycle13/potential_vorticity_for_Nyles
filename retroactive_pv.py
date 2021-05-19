import numpy as np
from netCDF4 import Dataset
import compiled_pv as cpv
import pickle


datadir = "/users/bertrandducrocq/data/Nyles"
expname = "ekman_2"
ncfile = "%s/%s/%s_00_hist.nc" % (datadir, expname, expname)

path = "%s/%s/param.pkl" % (datadir, expname)

with open(path, 'rb') as param:
    p = pickle.load(param)
    nz,ny,nx = p["nz"],p["ny"],p["nx"]
    nt = int(p["tend"]/p["timestep_history"])
    rotating = p["rotating"]
    coriolis = p["coriolis"]

#adding the coriolis force
if rotating == True:
    f = coriolis
else:
    f = 0.


pv = np.full((nz,ny,nx),0)


fn = "%s/%s/potential_vorticity.nc" % (datadir, expname)
with Dataset(fn, 'w', format='NETCDF4') as ds:
    time = ds.createDimension('time', None)
    x_dim = ds.createDimension('x', nx)
    y_dim = ds.createDimension('y', ny)
    z_dim = ds.createDimension('z', nz)

    stored_pv = ds.createVariable('potential_vorticity',np.float64, ('time','z','y','x'))

    for t in range(nt):

        with Dataset(ncfile, "r") as dc:
            b=dc.variables['b']
            w_i=dc.variables['vor_i']
            w_j=dc.variables['vor_j']
            w_k=dc.variables['vor_k']
            b = b[t,:,:,:]
            w_i = w_i[t,:,:,:]
            w_j = w_j[t,:,:,:]
            w_k = w_k[t,:,:,:]

        cpv.pv_computer(f,pv,b,w_i,w_j,w_k)
        stored_pv[t,:,:,:] = pv
