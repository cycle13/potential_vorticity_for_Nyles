"""

Projection functions to enforce div U = 0

"""
import numpy as np
import fortran_bernoulli as fortran
from timing import timing
import boundarycond as bc

localite = 0
pback = []

@timing
def compute_div(state, **kwargs):
    """Compute divergence from U, the contravariant velocity

    div = delta[U*vol]/vol

    here we assume that vol is uniform (Cartesian coordinates case)
    so

    div = delta[U]
    """
    for count, i in enumerate('ijk'):
        div = state.div.view(i)
        dU = state.U[i].view(i)
        fortran.div(div, dU, count)
        # if count == 0:
        #     div[:, :, 0] = dU[:, :, 0]
        #     div[:, :, 1:] = np.diff(dU)
        # else:
        #     div[:, :, 0] += dU[:, :, 0]
        #     div[:, :, 1:] += np.diff(dU)


@timing
def compute_p(mg, state, grid, ngbs):
    """
    This solves the poisson equation with U (state.U),
    stores the result in p (state.p)
    and corrects u (state.u) with -delta[p]

    note that the real pressure is p/dt

    mg is the multigrid object (with all data and methods)

    grid is the Grid object with the metric tensor
    """
    global localite, pback
    div = state.div
    compute_div(state)

    # at the end of the loop div and U are in the 'i' convention
    # this is mandatory because MG only works with the 'i' convention

    # copy divergence into the multigrid RHS
    # watch out, halo in MG is nh=1, it's wider for div
    b = mg.grid[0].b
    x = mg.grid[0].x

    # this is triplet of slices than span the MG domain (inner+MG halo)
    # typically mg_idx = (kidx, jidx, iidx)
    # with kidx = slice(k0, k1) the slice in the k direction
    mg_idx = state.div.mg_idx
    #idx = state.div.mg_idx
    #print(idx)
    d = div.view('i')
    #b[:] = div.view('i')[mg_idx]
    #l,m,n = np.shape(d)
    #ll,mm,nn = np.shape(b)
    #print(l,m,n,ll,mm,nn)
    mg.grid[0].toarray('b')
    mg.grid[0].toarray('x')
    #fortran.var2mg(d,b,idx,l,m,n,ll,mm,nn)
    #print(np.shape(b), mg_idx)
    b[:] = div.view('i')[mg_idx]
    mg.grid[0].tovec('b')
    mg.grid[0].tovec('x')

    p = state.p.view('i')
    # if localite>2:
    #     mg.grid[0].toarray('x')
    #     x[:] = pback[localite %3][mg_idx]
    #     mg.grid[0].tovec('x')
    # solve
    mg.solve_directly()

    # copy MG solution to pressure

    mg.grid[0].toarray('x')
    p[mg_idx] = x
    if localite>2:
        pback[localite%3][:] = p
    else:
        pback += [p.copy()]
    localite += 1
    #fortran.mg2var(p,x,idx)
    mg.grid[0].tovec('x')

    # correct u (the covariant component)
    # now we start with the 'i' convention
    for i in 'ijk':
        p = state.p.view(i)
        u = state.u[i].view(i)
        fortran.gradke(p, u)
        #u[:, :, :-1] -= np.diff(p)
