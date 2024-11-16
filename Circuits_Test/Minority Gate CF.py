"""Module for pipelines."""
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout, plot3d
import numpy as np
import matplotlib.pyplot as plt
import os



c = PCircuit(4)


#c.J = np.array([[0,-1,-1,2],[-1,0,-1,2],[-1,-1,0,2],[2,2,2,0]])
#c.h = np.array([0,0,0,1])


#NAND using minority Gate
#c.J = np.array([[0,1,1,-2],[1,0,1,-2],[1,1,0,-2],[-2,-2,-2,0]])
#c.h = np.array([2,0,0,0])


c.J = np.array([[0,-1,-1,-2],[-1,0,-1,-2],[-1,-1,0,-2],[-2,-2,-2,0]])

c.h = np.array([0,0,0,0])



solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

histplot(output)


#3d Histogram plot for the p-bit
#plot3d(output, A=[0,1,2], B=[3,4])
