#!/usr/bin/env python3

import numpy as np
import time
import sys
import os

# --------------------------------------------------
# FIX PATH FOR PYCHARM
# --------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from p_kit.solver.cuda_solver import CaSuDaSolver


# --------------------------------------------------
# GRAPH GENERATION
# --------------------------------------------------

def generate_erdos_renyi_graph(n_nodes, density=0.05, seed=None):

    if seed is not None:
        np.random.seed(seed)

    G = np.zeros((n_nodes, n_nodes), dtype=np.int32)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            if np.random.rand() < density:
                weight = np.random.randint(1, 10)

                G[i, j] = weight
                G[j, i] = weight

    n_edges = np.sum(G > 0) // 2

    return G, n_edges


# --------------------------------------------------
# MAXCUT -> ISING
# --------------------------------------------------

def maxcut_to_ising(G):

    J = -G.astype(np.float32) / 2.0

    h = np.zeros(G.shape[0], dtype=np.float32)

    return J, h


# --------------------------------------------------
# CUT CALCULATION
# --------------------------------------------------

def calculate_cut(G, sigma):

    n = len(sigma)

    cut = 0

    for i in range(n):
        for j in range(i + 1, n):

            if G[i, j] > 0 and sigma[i] != sigma[j]:
                cut += G[i, j]

    return cut


# --------------------------------------------------
# CREATE CIRCUIT
# --------------------------------------------------

def create_circuit(J, h):

    # YOUR SOLVER EXPECTS (J, h)

    return (J, h)


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def run_maxcut_experiment(
        n_nodes=50,
        n_trials=3,
        n_cycles=100,
        seed=42
):

    print("=" * 80)
    print("MAXCUT USING P-BIT SOLVER")
    print("=" * 80)

    # ----------------------------------------------
    # GENERATE GRAPH
    # ----------------------------------------------

    G, n_edges = generate_erdos_renyi_graph(
        n_nodes=n_nodes,
        density=0.05,
        seed=seed
    )

    print(f"Nodes : {n_nodes}")
    print(f"Edges : {n_edges}")

    # ----------------------------------------------
    # CONVERT TO ISING
    # ----------------------------------------------

    J, h = maxcut_to_ising(G)

    circuit = create_circuit(J, h)

    # ----------------------------------------------
    # MODES
    # ----------------------------------------------

    configs = [

        # {
        #     'name': 'Sequential Gibbs',
        #     'update_mode': 'sequential',
        #     'kwargs': {}
        # },

        {
            'name': 'CaSuDa',
            'update_mode': 'casuda',
            'kwargs': {}
        },

        {
            'name': 'pSA',
            'update_mode': 'psa',
            'kwargs': {}
        },

        {
            'name': 'TApSA',
            'update_mode': 'tapsa',
            'kwargs': {'alpha': 4}
        },

        {
            'name': 'SpSA',
            'update_mode': 'spsa',
            'kwargs': {'p_stall': 0.5}
        }
    ]
    # ----------------------------------------------
    # RUN
    # ----------------------------------------------

    for config in configs:

        print("\n" + "-" * 60)
        print(config['name'])
        print("-" * 60)

        solver = CaSuDaSolver(

            Nt=n_cycles,

            i0=3.0,
            i0_min=0.05,

            anneal=True,

            update_mode=config['update_mode'],

            backend='cupy',

            seed=seed,

            # IMPORTANT FIX
            record_samples=True,

            **config['kwargs']
        )

        cuts = []

        for trial in range(n_trials):

            start = time.time()

            energy_trace, samples = solver.solve(circuit)

            elapsed = time.time() - start

            # --------------------------------------
            # SAFETY FIX
            # --------------------------------------

            if len(samples) == 0:
                print("ERROR: samples are empty")
                continue

            final_sigma = samples[-1]

            final_cut = calculate_cut(G, final_sigma)

            cuts.append(final_cut)

            print(
                f"Trial {trial + 1} | "
                f"Cut = {final_cut} | "
                f"Time = {elapsed:.4f} sec"
            )

        if len(cuts) > 0:

            print("\nRESULTS")

            print(f"Mean Cut : {np.mean(cuts):.2f}")
            print(f"Best Cut : {np.max(cuts)}")
            print(f"Worst Cut: {np.min(cuts)}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--nodes',
        type=int,
        default=50
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=3
    )

    parser.add_argument(
        '--cycles',
        type=int,
        default=100
    )

    args = parser.parse_args()

    run_maxcut_experiment(
        n_nodes=args.nodes,
        n_trials=args.trials,
        n_cycles=args.cycles
    )