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

    #adding the coriolis force
    rotating = param.physics["rotating"]
    if rotating == True:
        f = param.physics["coriolis"]
    else:
        f = 0.

    cpv.pv_computer(f,pv,b,w_i,w_j,w_k)
