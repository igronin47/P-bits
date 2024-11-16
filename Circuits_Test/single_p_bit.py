
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout, plot3d
import numpy as np
import os

c = PCircuit(1)
c.J = np.array([[0]])
c.h = np.array([1])

solver = CaSuDaSolver(Nt=1200, dt=0.1667, i0=0.9, expected_mean=0.5)

input, output= solver.solve(c)

histplot(output)

np.set_printoptions(threshold=np.inf)


#3d Histogram plot for the p-bit
plot3d(output, A=[0], B=[])



