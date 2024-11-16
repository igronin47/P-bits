
from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
import os




c = PCircuit(4)


#c.J = np.array([[0,-1,-1,2],[-1,0,-1,2],[-1,-1,0,2],[2,2,2,0]])

#Bias of Majority Gate
#c.h = np.array([0,0,0,0])

#Bias of AND Gate using Majority Gate
#c.h = np.array([-2,0,0,0])

#Bias of OR Gate using Majority Gate
#c.h = np.array([2,0,0,0])


#Bias of 2:1 MUX using Majority Gate

#c.J = np.array([[0,-1,0,-1,2,0,0,0,0],[-1,0,-1,-1,2,-1,2,0,0],[0,-1,0,-1,0,-1,2,0,0],[-1,-1,-1,0,2,0,0,0,0],[2,2,0,2,0,0,-1,-1,2],[0,-1,-1,0,0,0,2,0,0],[0,2,2,0,-1,2,0,-1,2],[0,0,0,0,-1,0,-1,0,2],[0,0,0,0,2,0,2,2,0]])
#c.h = np.array([-2,-2,2,0,0,2,0,2,0])



solver = CaSuDaSolver(Nt=50000, dt=0.1667, i0=0.5)

input, output = solver.solve(c)

histplot(output)

#3d Histogram plot for the p-bit
plot3d(output, A=[0,1,2,3], B=[4,5,6,7])

