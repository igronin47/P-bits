from p_kit.core import PCircuit
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.visualization import histplot, vin_vout,plot3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


c = PCircuit(3)


c.J = np.array([[0,-1,2],[-1,0,2],[2,2,0]])
c.h = np.array([-1,-1,2])


solver = CaSuDaSolver(Nt=25000, dt=0.1667, i0=0.9)

input, output = solver.solve(c)

histplot(output)


#np.set_printoptions(threshold=np.inf)
#print(output)



plot3d(output, A=[0,1], B=[2])

