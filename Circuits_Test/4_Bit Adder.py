
import sys
import os
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt



c = PCircuit(21)

# 4-Bit Adder using FA

c = PCircuit(21)

c.J = np.array([  [0,0,0,3,-3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,-2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,-2,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [3,2,2,0,1,-2,0,-2,2,2,0,0,0,0,0,0,0,0,0,0,0],
                  [-3,2,2,1,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [2,0,0,-2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,3,-3,2,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,-2,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,2,0,0,3,2,0,1,-2,0,-2,2,2,0,0,0,0,0,0],
                  [0,0,0,2,0,0,-3,2,1,0,2,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,2,0,-2,2,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,3,-3,2,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,-2,0,0,0,0,2,2,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,2,0,0,3,2,0,1,-2,0,-2,2,2,0],
                  [0,0,0,0,0,0,0,0,2,0,0,-3,2,1,0,2,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,2,0,-2,2,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,-3,2],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,2,2,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,3,2,0,1,-2],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,-3,2,1,0,2],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,-2,2,0]])


c.h = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#for clamping p-bits nodes are given, use bias matrices to clamp
# input A=[1,7,12,17]
# input B=[2,8,13,18]
# Cin = [3]
# output =[6,11,16,21,19]



solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

histplot(output)
