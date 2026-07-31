from p_kit.core import PCircuit
from p_kit.visualization import histplot
from p_kit.solver.cuda_solver import CudaSolver

import numpy as np
import matplotlib.pyplot as plt
import time
c = PCircuit(4)


c.J = np.array([[0,-1,-1,2],
                [-1,0,-1,2],
                [-1,-1,0,2],
                [2,2,2,0]]
               , dtype=np.float32)

#Bias of Majority Gate
c.h = np.array([0,0,0,0], dtype=np.float32)



modes = ["sequential", "psa", "tapsa", "spsa"]

results = {}

# ============================================================
# Run All Modes
# ============================================================

for mode in modes:

    print("\n======================================")
    print("Running Mode:", mode)
    print("======================================")

    t0 = time.time()

    solver = CudaSolver(
        Nt=100000,
        dt=0.1667,
        i0=0.5,
        update_mode=mode,
        backend="numpy",
        alpha=4,          # Used only for tapsa
        p_stall=0.5       # Used only for spsa
    )

    # backend = "numpy"
    # backend = "cupy"
    # backend = "cuda"
    #12 execution paths
    #4 solver modes × 3 backends

    print("Backend:", solver.backend)




    energies, output = solver.solve(c)

    runtime = time.time() - t0

    print("Runtime:", runtime, "seconds")
    print("Final Energy:", energies[-1])

    results[mode] = {
        "energies": energies,
        "output": output,
        "runtime": runtime
    }

    # print("Plotting Histogram...")
    # histplot(output)

# ============================================================
# Compare Energy Convergence
# ============================================================

plt.figure(figsize=(10,6))

for mode in modes:
    plt.plot(results[mode]["energies"], label=mode)

# plt.xlabel("Iteration")
# plt.ylabel("Energy")
# plt.title("Energy Comparison of Different Solver Modes")
# plt.legend()
# plt.grid(True)
#
# plt.show()

# ============================================================
# Runtime Comparison
# ============================================================

print("\n================ RUNTIME SUMMARY ================\n")

for mode in modes:
    print(f"{mode:12s} : {results[mode]['runtime']:.4f} seconds")