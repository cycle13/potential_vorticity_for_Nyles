import vortex_force as vortf
import variables as var
import tracer
import timescheme as ts
import vorticity as vort
import bernoulli as bern
import kinenergy as kinetic
import viscosity as visc
import projection
import boundarycond as bc
import topology as topo
from timing import timing
import mg
import pickle
import cov_to_contra
import halo
import potential_vorticity
import parameters


import numpy as np
import matplotlib.pyplot as plt

"""
LES model

At each step n, this model doesn't suppose that the velocity at step n-1 had 0 divergence.
It follows the procedure from the Ferziger p.180

"""


class LES(object):

    def __init__(self, param, grid, linear=False):
        self.param = param
        self.nonlinear = not linear
        self.grid = grid
        self.traclist = ['b']
        # Add tracers (if any)
        for i in range(param["n_tracers"]):
            t_nickname = "t{}".format(i)
            t_name = "tracer{}".format(i)
            self.traclist.append(t_nickname)
            var.modelvar[t_nickname] = var.ModelVariable(
                'scalar', t_name,  dimension='', prognostic=True)
        self.state = var.get_state(param)
        self.halo = halo.set_halo(param, self.state)
        self.neighbours = param["neighbours"]
        self.timescheme = ts.Timescheme(param, self.state)
        self.timescheme.set(self.rhs, self.diagnose_var)
        self.orderA = param["orderA"]
        self.orderVF = param["orderVF"]
        self.orderKE = param["orderKE"]
        self.rotating = param["rotating"]
        self.forced = param["forced"]
        self.diff_coef = param['diff_coef']
        self.add_viscosity = "u" in self.diff_coef.keys()
        if self.add_viscosity:
            self.viscosity = self.diff_coef['u']

        self.tracer = tracer.Tracer_numerics(param,
            grid, self.traclist, self.orderA, self.diff_coef)
        if self.rotating:
            # convert Coriolis parameter (in s^-1) into its covariant quantity
            # i.e. multiply with cell horizontal area
            area = self.grid.dx*self.grid.dy
            self.fparameter = param["coriolis"] * area
        else:
            self.fparameter = 0.
        self.mg = mg.Multigrid(param, grid)
        self.stats = []

    @timing
    def diagnose_var(self, state):
        #bc.apply_bc_on_velocity(state, self.neighbours)
        self.halo.fill(state.b)
        self.halo.fill(state.u)

        # Diagnostic variables
        cov_to_contra.U_from_u(state, self.grid)
        projection.compute_p(self.mg, state, self.grid, self.neighbours)
        self.halo.fill(state.u)
        cov_to_contra.U_from_u(state, self.grid)
        potential_vorticity.pv_computer(state,self.param)


        # this computation is only to check the divergence
        # after the projection, this could be drop if
        # we don't need to know the information (that can
        # always be estimated offline, from 'u')
        projection.compute_div(self.state, timing=False)
        self.halo.fill(state.div)
        self.update_stats()

        if self.nonlinear:
            vort.vorticity(state, self.fparameter)
            #bc.apply_bc_on_vorticity(state, self.neighbours)
            kinetic.kinenergy(state, self.grid, self.orderKE)
            self.halo.fill(state.vor)
            self.halo.fill(state.ke)





    @timing
    def rhs(self, state, t, dstate, last=False):
        reset_state(dstate)
        # transport the tracers
        self.tracer.rhstrac(state, dstate, last=last)
        # vortex force
        if self.nonlinear:
            vortf.vortex_force(state, dstate, self.orderVF)
        # bernoulli
        bern.bernoulli(state, dstate, self.grid)


        if self.forced and last:
            self.forcing.add(state, dstate, t)

        if self.add_viscosity and last:
            visc.add_viscosity(self.grid, state, dstate, self.viscosity)


    @timing
    def forward(self, t, dt):
        self.timescheme.forward(self.state, t, dt)
        return self.mg.stats['blowup']

    def update_stats(self):
        stats = self.mg.stats

        div = self.state.div
        maxdiv = np.max(np.abs(div.view()))
        stats['maxdiv'] = maxdiv
        if hasattr(self, 'stats'):
            self.stats += [stats]
        else:
            self.stats = [stats]

    def write_stats(self, path):
        fid = open('%s/stats.pkl' % path, 'bw')
        pickle.dump(self.stats, fid)


def reset_state(state):
    for var_name, var_type in state.toc.items():
        if var_type == "scalar":
            var = state.get(var_name).view()
            var[...]= 0.0
        else:
            for i in "ijk":
                var = state.get(var_name)[i].view()
                var[...]= 0.0
