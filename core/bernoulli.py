"""Module to compute the bernoulli term without pressure.

TODO:
  to increase computational intensity, compute delta[b*grad(z) - grad(ke)]
  and add it to the rhs of the multigrid

"""

import fortran_bernoulli as fortran
from timing import timing


@timing
def bernoulli(state, rhs, grid):
    """Add b*grad(z)-grad(ke) to the rhs."""
    for i in 'ijk':
        du_i = rhs.u[i].view(i)
        ke = state.ke.view(i)
        if i in 'ij':
            fortran.gradke(ke, du_i)
        else:
            b = state.b.view(i)
            fortran.gradkeandb(ke, b, du_i, grid.dz)
