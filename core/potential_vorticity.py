import compiled_pv as cpv
import numpy as np
import parameters as p

def pv_computer(state, f):

    param = p.UserParameters()

    b = state.b.view('i')
    pv = state.pv.view('i')
    w_i = state.vor["i"].view('i')
    w_j = state.vor["j"].view('i')
    w_k = state.vor["k"].view('i')

    av_gradb_i = np.full_like(b,0)
    av_gradb_j = np.full_like(b,0)
    av_gradb_k = np.full_like(b,0)

    av_w_i = np.full_like(b,0)
    av_w_j = np.full_like(b,0)
    av_w_k = np.full_like(b,0)
    """
    #adding the coriolis force
    rotating = param.physics["rotating"]
    if rotating == False:
        f = 0

    w_k[:, :-1, :-1] += f
    """
    cpv.pv_computer(pv,b,w_i,w_j,w_k,av_gradb_i,av_gradb_j,av_gradb_k,av_w_i,av_w_j,av_w_k)
