from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot
import numpy as np



# =========================================================
# 2-to-4 DECODER
# Variables: [x1, x2, y0, y1, y2, y3]
# =========================================================

    c.J = np.array([
        [ 0,  0, -2,  2, -2,  2],   # x1
        [ 0,  0, -2, -2,  2,  2],   # x2
        [-2, -2,  0,  0,  0,  0],   # y0
        [ 2, -2,  0,  0,  0,  0],   # y1
        [-2,  2,  0,  0,  0,  0],   # y2
        [ 2,  2,  0,  0,  0,  0]    # y3
    ])

    c.h = np.zeros(6)


solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)
