# %%
import os
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
import python_utils.utils as utils
import python_utils.simulation as simulation

# %%
# import matplotlib
# import matplotlib.pyplot as plt

# # %%
# import matplotlib

# matplotlib.use("pgf")
# import matplotlib.pyplot as plt

# plt.rcParams.update(
#     {
#         "font.family": "serif",  # use serif/main font for text elements
#         "text.usetex": True,  # use inline math for ticks
#         "pgf.rcfonts": False,  # don't setup fonts from rc parameters
#         "font.size": 9,
#     }
# )


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
        import admm_fq_distavg_opt as adf
    if alg == "davgad":
        import admm_fq_distavg_ad as adf
    from cvxpy import Variable, sum_squares, Minimize, Problem, vec

    rng = np.random.default_rng(np.random.PCG64DXSM(seed).jumped(run))
    ## %%
    # ################################################################
    # Put the simulation code here
    N_sens = 20
    L = 16

    nw = adf.Network(L)

    next_pos = np.array([0, 0])
    node_pos = np.empty((N_sens, 2))
    # fig = plt.figure(figsize=(8, 8))
    nbs = []
    ok = []
    edges = []
    radius = 2
    for i in range(N_sens):
        nw.addNode(i, 1)
        rangle = rng.uniform(-np.pi / 4, np.pi / 4) + rng.integers(0, 2) * np.pi
        # print(rangle)
        # rd = rng.uniform(0.8 * radius, 0.99*radius)
        rd = radius * 0.95
        next_pos = next_pos + np.array([rd * np.cos(rangle), rd * np.sin(rangle)])
        node_pos[i, :] = next_pos

    for i in range(N_sens):
        d = np.sqrt(np.sum(np.square(node_pos[i, :] - node_pos), 1))
        nb_pos = node_pos[d <= radius, :]
        nb_ind = (d <= radius).nonzero()[0]
        if len(nb_ind) > 1:
            ok.append(i)
            nw.setConnection(i, nb_ind)
            for j in nb_ind:
                if (i, j) not in edges and (j, i) not in edges:
                    edges.append((i, j))

    # node_posa = np.asarray(node_pos)
    edge_arr = np.asarray(edges)
    x = [node_pos[edge_arr[:, 0], 0], node_pos[edge_arr[:, 1], 0]]
    y = [node_pos[edge_arr[:, 0], 1], node_pos[edge_arr[:, 1], 1]]
    # plt.plot(x, y, "-", color="tab:gray", linewidth=0.75)
    # plt.scatter(node_pos[ok, 0], node_pos[ok, 1], c="k")

    if alg != "opt":
        connectivity_matrix_reshaped = nw.A_.reshape(nw.N * nw.N, 1)
        pp = np.where(connectivity_matrix_reshaped == 0)
        connectivity_matrix_zero_indices = pp[0].reshape(-1, 1)

        W_opt = Variable((nw.N, nw.N))
        objective = Minimize(
            sum_squares(
                W_opt - np.divide((np.ones((nw.N, 1)) * np.ones((1, nw.N))), nw.N)
            )
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
        nw.setAveragingWeights(W)

    rho = 1.0
    stepsize = 0.5
    eta = 0.98
    N_sens = nw.N
    N_s = 50000

    # true_norms = [2.2, 0.5, 1.2, 0.7, 1.0]
    true_norms = [rng.uniform(0.5, 2.0) for i in range(N_sens)]

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
    # cfg = simulation.SimConfig(
    #     id="icassp2023-static-large",
    #     runs=30,
    #     seed=573438,
    #     variables=[
    #         {"alg": ["opt"], "SNR": [10], "gamma": [0.0], "iters": [1]},
    #         {"alg": ["davgad"], "SNR": [10], "gamma": [0.0], "iters": [1]},
    #         {
    #             "alg": ["davg"],
    #             "SNR": [10],
    #             "gamma": [
    #                 0.0,
    #                 0.002,
    #                 0.004,
    #                 0.006,
    #                 0.008,
    #                 0.01,
    #                 0.012,
    #                 0.014,
    #                 0.016,
    #                 0.018,
    #                 0.02,
    #                 0.025,
    #                 0.03,
    #                 0.04,
    #             ],
    #             "iters": [1, 10],
    #         },
    #     ],
    # )
    # %%
    # You can save the simulation configuration to a json file
    # cfg.save("icassp2023-static-large")
    # %%
    # Load a saved configuration
    cfg = simulation.SimConfig.load("icassp2023-static-large")
    # %%
    # Create the simulation instance. (dont change the function parameters here pls)
    sim = simulation.Simulation(cfg, os.path.realpath(__file__), simulate)
    # %%
    # Delete generated data if necessary
    # sim.clearTmpData()
    # %%
    # Run the simulation locally with n processes
    sim.runLocal(nprocesses=3, showprogress=False)
    # %%
    # Get the result after completion
    # result = sim.getResult()
    # %%
    # Define condor job parameters
    # user_submit = {
    #     "request_walltime": "3600",  # Time in seconds
    #     "initialdir": ".",
    #     "notification": "Error",
    #     "executable": "/users/sista/mblochbe/python_venvs/admmstuff/bin/python",
    #     "request_cpus": "1",
    #     "request_memory": "1GB",
    # }
    # # %%
    # # Submit the condor jobs
    # sim.runCondor(user_submit)
    # # %%
    # # Show status of condor jobs
    # sim.getJobStatus()
    # # %%
    # # Check if all are done
    # print(sim.isDone())
    # # %%
    # # Get result if done
    # if sim.isDone():
    #     result = sim.getResult()
    # # %%
    # data = result.df.groupby(["alg", "gamma", "iters"])

    # # %%
    # textwidth = 245
    # conv_frames = 500
    # styles = ["-v", "-+", "-x", "-s", "-o", "k-"]
    # fig = plt.figure(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.5))
    # # plt.title(f"SNR={cfg.variables['SNR'][0]}dB")
    # plt.xlabel(r"Mixing factor $\gamma_i$ [1]")
    # plt.ylabel("Avg. NPM [dB]")
    # plt.plot(
    #     cfg.variable_values[2]["gamma"],
    #     20
    #     * np.log10(data.median().T.tail(conv_frames).mean()["opt", 1.0, 1])
    #     * np.ones_like(cfg.variable_values[2]["gamma"]),
    #     styles.pop(),
    #     label=f"optimal",
    #     markersize=4,
    #     markevery=(1, 2),
    #     markerfacecolor="none",
    # )
    # plt.plot(
    #     cfg.variable_values[2]["gamma"],
    #     20
    #     * np.log10(data.median().T.tail(conv_frames).mean()["davgad", 1.0, 1])
    #     * np.ones_like(cfg.variable_values[2]["gamma"]),
    #     styles.pop(),
    #     label=rf"adaptive $\gamma_i,\,K=1$",
    #     markersize=4,
    #     markevery=(2, 2),
    #     markerfacecolor="none",
    # )
    # plt.plot(
    #     cfg.variable_values[2]["gamma"],
    #     20 * np.log10(data.median().T.tail(conv_frames).mean()["davg", :, 1]),
    #     styles.pop(),
    #     label=rf"fixed $\gamma_i,\,K=1$",
    #     markersize=4,
    #     markevery=(1, 2),
    #     markerfacecolor="none",
    # )
    # plt.plot(
    #     cfg.variable_values[2]["gamma"],
    #     20 * np.log10(data.median().T.tail(conv_frames).mean()["davg", :, 2]),
    #     styles.pop(),
    #     label=rf"fixed $\gamma_i,\,K=2$",
    #     markersize=4,
    #     markevery=(2, 2),
    #     markerfacecolor="none",
    # )
    # plt.plot(
    #     cfg.variable_values[2]["gamma"],
    #     20 * np.log10(data.median().T.tail(conv_frames).mean()["davg", :, 10]),
    #     styles.pop(),
    #     label=rf"fixed $\gamma_i,\,K=10$",
    #     markersize=4,
    #     markevery=(1, 2),
    #     markerfacecolor="none",
    # )

    # plt.grid()
    # plt.ylim(-40, -20)
    # plt.xlim(0, 0.04)
    # plt.legend(ncol=2, prop={"size": 7}, columnspacing=0.5)
    # plt.tight_layout(pad=0.5)
    # plt.show()

    # ax = plt.gca()
    # box = ax.get_position()

    # # %%
    # utils.savefig(fig, "icassp2023-static-large", format="pgf", pgf_font="serif")

    # # %%
    # # Plot
    # mavg = 200
    # styles = ["-+", "-x", "-<", "->", "-v", "-s", "-o", "k-"]
    # fig = plt.figure(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.5))
    # # plt.title("Title")
    # plt.xlabel("Time [frames]")
    # plt.ylabel("NPM [dB]")
    # # #########
    # plt.fill_between(
    #     [6000 - conv_frames, 6000],
    #     [-40, -40],
    #     [0, 0],
    #     facecolor="tab:gray",
    #     hatch="//",
    #     edgecolor="k",
    #     linewidth=0.5,
    #     alpha=0.5,
    # )
    # (line,) = plt.plot(
    #     20 * np.log10(data.median().T["opt", 1.0, 1]),
    #     "k-",
    #     label=f"_optimal",
    #     markersize=4,
    #     markevery=(1, 500),
    #     alpha=0.25,
    # )
    # plt.plot(
    #     20
    #     * np.log10(
    #         np.convolve(
    #             data.median().T["opt", 1.0, 1], np.ones(mavg) / mavg, mode="valid"
    #         )
    #     ),
    #     "-",
    #     label=f"optimal",
    #     markersize=4,
    #     markevery=(1, 500),
    #     color=line.get_color(),
    #     markerfacecolor="none",
    # )
    # # #########
    # (line,) = plt.plot(
    #     20 * np.log10(data.median().T["davgad", 1.0, 1]),
    #     "-",
    #     label=rf"_adaptive $\gamma_i,\,K=1$",
    #     markersize=4,
    #     markevery=(20, 500),
    #     alpha=0.25,
    # )
    # plt.plot(
    #     20
    #     * np.log10(
    #         np.convolve(
    #             data.median().T["davgad", 1.0, 1], np.ones(mavg) / mavg, mode="valid"
    #         )
    #     ),
    #     "-o",
    #     label=rf"adaptive $\gamma_i,\,K=1$",
    #     markersize=4,
    #     markevery=(20, 500),
    #     color=line.get_color(),
    #     markerfacecolor="none",
    # )
    # # #########
    # (line,) = plt.plot(
    #     20 * np.log10(data.median().T["davg", 0.01, 1]),
    #     "-",
    #     label=rf"_$\gamma_i = 0.01,\,K=1$",
    #     markersize=4,
    #     markevery=(40, 500),
    #     alpha=0.25,
    # )
    # plt.plot(
    #     20
    #     * np.log10(
    #         np.convolve(
    #             data.median().T["davg", 0.01, 1], np.ones(mavg) / mavg, mode="valid"
    #         )
    #     ),
    #     "-s",
    #     label=rf"$\gamma_i = 0.01,\,K=1$",
    #     markersize=4,
    #     markevery=(40, 500),
    #     color=line.get_color(),
    #     markerfacecolor="none",
    # )
    # # #########
    # (line,) = plt.plot(
    #     20 * np.log10(data.median().T["davg", 0.01, 10]),
    #     "-",
    #     label=rf"_$\gamma_i = 0.01,\,K=10$",
    #     markersize=4,
    #     markevery=(60, 500),
    #     alpha=0.25,
    # )
    # plt.plot(
    #     20
    #     * np.log10(
    #         np.convolve(
    #             data.median().T["davg", 0.01, 10], np.ones(mavg) / mavg, mode="valid"
    #         )
    #     ),
    #     "-x",
    #     label=rf"$\gamma_i = 0.01,\,K=10$",
    #     markersize=4,
    #     markevery=(60, 500),
    #     color=line.get_color(),
    #     markerfacecolor="none",
    # )
    # # #########
    # (line,) = plt.plot(
    #     20 * np.log10(data.median().T["davg", 0.04, 1]),
    #     "-",
    #     label=rf"_$\gamma_i = 0.04,\,K=1$",
    #     markersize=4,
    #     markevery=(80, 500),
    #     alpha=0.25,
    # )
    # plt.plot(
    #     20
    #     * np.log10(
    #         np.convolve(
    #             data.median().T["davg", 0.04, 1], np.ones(mavg) / mavg, mode="valid"
    #         )
    #     ),
    #     "-+",
    #     label=rf"$\gamma_i = 0.04,\,K=1$",
    #     markersize=4,
    #     markevery=(80, 500),
    #     color=line.get_color(),
    #     markerfacecolor="none",
    # )
    # # #########
    # (line,) = plt.plot(
    #     20 * np.log10(data.median().T["davg", 0.04, 10]),
    #     "-",
    #     label=rf"_$\gamma_i = 0.04,\,K=10$",
    #     markersize=4,
    #     markevery=(100, 500),
    #     alpha=0.25,
    # )
    # plt.plot(
    #     20
    #     * np.log10(
    #         np.convolve(
    #             data.median().T["davg", 0.04, 10], np.ones(mavg) / mavg, mode="valid"
    #         )
    #     ),
    #     "-v",
    #     label=rf"$\gamma_i = 0.04,\,K=10$",
    #     markersize=4,
    #     markevery=(100, 500),
    #     color=line.get_color(),
    #     markerfacecolor="none",
    # )
    # # #########
    # plt.legend(ncol=2, prop={"size": 7}, columnspacing=0.5)
    # plt.grid()
    # plt.ylim(-40, 0)
    # plt.xlim(0, 6000)
    # plt.tight_layout(pad=0.5)
    # ax = plt.gca()
    # ax.set_position(box)
    # plt.show()
    # # %%
    # utils.savefig(fig, "icassp2023-static-large-time", format="pgf", pgf_font="serif")

# %%
