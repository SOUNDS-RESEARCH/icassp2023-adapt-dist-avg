# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import python_utils.utils as utils
import python_utils.simulation as simulation
from cvxpy import Variable, sum_squares, Minimize, Problem, vec


# %%
# Define the simulation function.
# You can define any number of arguments instead of a, b.
# Remember to add them in SimConfig
# DO NOT CHANGE run and seed arguments and argument positions!
def simulate(alg, SNR, gamma, iters, run: int, seed: int):
    ## %%
    # All required imports for simulation need to be in here, e.g.,
    import numpy as np

    if alg == "opt":
        import admm_fq as adf
    if alg == "davg":
        import admm_fq_distavg as adf

    rng = np.random.default_rng(np.random.PCG64DXSM(seed).jumped(run))
    ## %%
    # ################################################################
    # Put the simulation code here
    connectivity_matrix_reshaped = nw.A_.reshape(nw.N * nw.N, 1)
    pp = np.where(connectivity_matrix_reshaped == 0)
    connectivity_matrix_zero_indices = pp[0].reshape(-1, 1)

    W_opt = Variable((nw.N, nw.N))
    objective = Minimize(
        sum_squares(W_opt - np.divide((np.ones((nw.N, 1)) * np.ones((1, nw.N))), nw.N))
    )
    W_vec = vec(W_opt)
    constraints = (
        W_vec[connectivity_matrix_zero_indices] == 0,
        np.ones([1, nw.N]) @ W_opt == np.ones([1, nw.N]),
        W_opt @ np.ones([nw.N, 1]) == np.ones([nw.N, 1]),
    )
    prob = Problem(objective, constraints)
    result = prob.solve()
    W = W_opt.value.copy()

    L = 16
    nw = adf.Network(L)
    nw.addNode(0, 1)
    nw.addNode(1, 1)
    nw.addNode(2, 1)
    nw.addNode(3, 1)
    nw.addNode(4, 1)
    nw.setConnection(0, [1])
    nw.setConnection(1, [2])
    nw.setConnection(2, [3])
    nw.setConnection(3, [4])
    nw.setConnection(4, [0])

    W = np.array(
        [
            [0.5, 0.0, 0.0, 0.0, 0.5],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.5],
        ]
    )
    if alg != "opt":
        nw.setAveragingWeights(W)

    rho = 1.0
    stepsize = 0.8
    eta = 0.98
    N_sens = nw.N
    N_s = 80000

    # true_norms = [2.2, 0.5, 1.2, 0.7, 1.0]
    true_norms = [rng.uniform(0.5, 3.0) for i in range(N_sens)]

    h = np.array([]).reshape(0, 1)
    for n in range(N_sens):
        h_ = rng.normal(size=(L, 1))
        h_ = h_ / np.linalg.norm(h_) * true_norms[n]
        h = np.concatenate([h, h_])

    u = rng.normal(size=(N_s, 1))
    s = u / u.max()

    hopsize = L
    s_ = np.concatenate([np.zeros(shape=(L - 1, 1)), s])
    var_s = np.var(s)

    x_ = np.zeros(shape=(N_s, N_sens))
    n_var = 10 ** (-SNR / 20) * var_s * np.linalg.norm(h) ** 2 / N_sens
    H = np.reshape(h, (N_sens, L))
    for k in range(N_s - L):
        x_[k, :, None] = H @ s_[k : k + L][::-1] + n_var * rng.normal(size=(N_sens, 1))
    x_x = x_

    hopsize = L

    data = []
    h_test = np.zeros(shape=h.shape)
    nw.reset()
    if alg == "opt":
        nw.setRho(rho, stepsize, eta, 1)
    else:
        nw.setRho(rho, stepsize, eta, 1, iters, gamma)
    for k_admm_fq in range(0, N_s - 2 * L, hopsize):
        nw.step(x_x[k_admm_fq : k_admm_fq + 2 * L, :])
        for n in range(N_sens):
            h_test[(n * L) : (n + 1) * L] = np.real(
                np.fft.ifft(nw.z[(n * L) : (n + 1) * L], axis=0)
            )
        e1 = h - (h.T @ h_test) / (h_test.T @ h_test) * h_test
        error = np.linalg.norm(e1) / np.linalg.norm(h)
        data.append(error)
    # ################################################################
    ## %%
    return data


# %%
if __name__ == "__main__":  # Necessary for module loading in condor processes :(
    cfg = simulation.SimConfig(
        id="icassp2023-static",
        runs=30,
        seed=573438,
        variables=[
            {"alg": ["opt"], "SNR": [10], "gamma": [1.0], "iters": [1]},
            {
                "alg": ["davg"],
                "SNR": [10],
                "gamma": [
                    0.0,
                    0.002,
                    0.004,
                    0.006,
                    0.008,
                    0.01,
                    0.012,
                    0.014,
                    0.016,
                    0.018,
                    0.02,
                ],
                "iters": [1, 2, 10],
            },
        ],
    )
    # %%
    # You can save the simulation configuration to a json file
    cfg.save("icassp2023-static")
    # %%
    # Load a saved configuration
    cfg = simulation.SimConfig.load("icassp2023-static")
    # %%
    # Create the simulation instance. (dont change the function parameters here pls)
    sim = simulation.Simulation(cfg, os.path.realpath(__file__), simulate)
    # %%
    # Delete generated data if necessary
    sim.clearTmpData()
    # %%
    # Run the simulation locally with n processes
    sim.runLocal(nprocesses=4, showprogress=True)
    # %%
    # Get the result after completion
    result = sim.getResult()
    # %%
    # Define condor job parameters
    user_submit = {
        "request_walltime": "86000",  # Time in seconds
        "initialdir": ".",
        "notification": "Error",
        "executable": "/users/sista/mblochbe/python_venvs/admmstuff/bin/python",
        "request_cpus": "1",
        "request_memory": "8GB",
    }
    # %%
    # Submit the condor jobs
    sim.runCondor(user_submit)
    # %%
    # Show status of condor jobs
    sim.getJobStatus()
    # %%
    # Check if all are done
    print(sim.isDone())
    # %%
    # Get result if done
    if sim.isDone():
        result = sim.getResult()
    # %%
    data = result.df.groupby(["alg", "SNR", "gamma", "iters"])
    # %%
    # Plot
    fig = plt.figure(figsize=(6, 4))
    plt.title("Title")
    plt.xlabel("Series [1]")
    plt.ylabel("Value [1]")
    labels = [rf"{label}" for label in data.groups.keys()]
    plt.plot(
        20 * np.log10(data.median().to_numpy().T),
        label=labels,
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    # %%
    utils.savefig(fig, "icassp2023-static_plot", format="png")
