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

# %%
import matplotlib

matplotlib.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "font.size": 9,
    }
)


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

    rng = np.random.default_rng(np.random.PCG64DXSM(seed).jumped(run))
    ## %%
    # ################################################################
    # Put the simulation code here
    L = 16
    nw = adf.Network(L)
    nw.addNode(0, 1)
    nw.addNode(1, 1)
    nw.addNode(2, 1)
    nw.setConnection(0, [1])
    nw.setConnection(1, [2])
    nw.setConnection(2, [0])

    W = np.array(
        [
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    if alg != "opt":
        nw.setAveragingWeights(W)

    rho = 1.0
    stepsize = 0.5
    eta = 0.98
    N_sens = nw.N
    N_s = 80000
    SNR = 10
    iters = 1
    gamma = 0.000

    h_s = []
    for n in range(N_sens):
        h_ = rng.normal(size=(L, 1))
        h_ = h_ / np.linalg.norm(h_)
        h_s.append(h_)

    x_x = np.array([]).reshape(0, N_sens)
    true_norms = [[2.2, 0.5, 1.2], [2.2, 1.0, 1.2], [2.2, 0.5, 2.0]]
    # true_norms = [rng.uniform(0.5, 2.0) for i in range(N_sens)]
    for true in true_norms:
        h = np.array([]).reshape(0, 1)
        for n in range(N_sens):
            h_ = h_s[n] * true[n]
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
            x_[k, :, None] = H @ s_[k : k + L][::-1] + n_var * rng.normal(
                size=(N_sens, 1)
            )
        # x_x.append(x_)
        x_x = np.concatenate([x_x, x_])

    hopsize = L

    npm_error = []
    h_test = np.zeros(shape=h.shape)
    nw.reset()
    if alg == "opt":
        nw.setRho(rho, stepsize, eta, 1)
    else:
        nw.setRho(rho, stepsize, eta, 1, iters, gamma)
    for k_admm_fq in range(0, len(true_norms) * N_s - 2 * L, hopsize):
        nw.step(x_x[k_admm_fq : k_admm_fq + 2 * L, :])
        error = 0
        for n in range(N_sens):
            h_test = np.real(np.fft.ifft(nw.z[(n * L) : (n + 1) * L], axis=0))
            e1 = (
                h[(n * L) : (n + 1) * L]
                - (h[(n * L) : (n + 1) * L].T @ h_test) / (h_test.T @ h_test) * h_test
            )
            error += np.square(
                np.linalg.norm(e1) / np.linalg.norm(h[(n * L) : (n + 1) * L])
            )
        npm_error.append(np.sqrt(error))
    if alg != "opt":
        norms_t = np.asarray(nw.norms_t).squeeze()
        data = np.concatenate(
            [npm_error, nw.norm_t, norms_t[:, 0], norms_t[:, 1], norms_t[:, 2]]
        )
    else:
        data = np.concatenate([npm_error, np.zeros((len(npm_error) * 4,))])
    # ################################################################
    ## %%
    return data


# %%
if __name__ == "__main__":  # Necessary for module loading in condor processes :(
    cfg = simulation.SimConfig(
        id="icassp2023-dynamic",
        runs=30,
        seed=573438,
        variables=[
            {"alg": ["davgad", "opt"], "SNR": [10], "gamma": [0.0], "iters": [1]},
        ],
    )
    # %%
    # You can save the simulation configuration to a json file
    cfg.save("icassp2023-dynamic")
    # %%
    # Load a saved configuration
    cfg = simulation.SimConfig.load("icassp2023-dynamic")
    # %%
    # Create the simulation instance. (dont change the function parameters here pls)
    sim = simulation.Simulation(cfg, os.path.realpath(__file__), simulate)
    # %%
    # Delete generated data if necessary
    # sim.clearTmpData()
    # %%
    # Run the simulation locally with n processes
    sim.runLocal(nprocesses=8, showprogress=True)
    # %%
    # Get the result after completion
    result = sim.getResult()
    # %%
    # Define condor job parameters
    user_submit = {
        "request_walltime": "3600",  # Time in seconds
        "initialdir": ".",
        "notification": "Error",
        "executable": "/users/sista/mblochbe/python_venvs/admmstuff/bin/python",
        "request_cpus": "1",
        "request_memory": "1GB",
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
    frames = int(result.df.shape[1] / 5)
    data = result.df.groupby(["alg", "gamma", "iters"])

    # %%
    # Plot
    textwidth = 245
    linewidth = 1.2
    plt.close("all")
    mavg = 10
    styles = ["-+", "-x", "-<", "->", "-v", "-s", "-o", "k-"]
    fig = plt.figure(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.4))
    # plt.title("Title")
    plt.xlabel("Time [frames]")
    plt.ylabel("NPM [dB]")
    plt.vlines(
        [5000, 10000],
        -30,
        0,
        colors=["k", "k"],
        linestyles=["dashed", "dashed"],
        linewidth=[0.75, 0.75],
    )
    (line,) = plt.plot(
        20 * np.log10(data.median().T["opt", 0.0, 1][:frames]),
        "k-",
        label=rf"_optimal",
        markersize=4,
        markevery=(20, 500),
        alpha=0.25,
        linewidth=linewidth,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["opt", 0.0, 1][:frames],
                np.ones(mavg) / mavg,
                mode="valid",
            )
        ),
        "-",
        label=rf"optimal",
        markersize=4,
        markevery=(20, 500),
        color=line.get_color(),
        markerfacecolor="none",
        linewidth=linewidth,
    )
    (line,) = plt.plot(
        20 * np.log10(data.median().T["davgad", 0.0, 1][:frames]),
        "-",
        label=rf"_adaptive $\gamma_i,\,K=1$",
        markersize=4,
        markevery=(20, 500),
        alpha=0.25,
        linewidth=linewidth,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["davgad", 0.0, 1][:frames],
                np.ones(mavg) / mavg,
                mode="valid",
            )
        ),
        "-o",
        label=rf"adaptive $\gamma_i,\,K=1$",
        markersize=4,
        markevery=(20, 500),
        color=line.get_color(),
        markerfacecolor="none",
        linewidth=linewidth,
    )
    plt.legend(ncol=2, prop={"size": 7}, columnspacing=0.5, loc="upper center")
    plt.grid()
    plt.ylim(-30, 0)
    plt.xlim(0, 15000)
    plt.tight_layout(pad=0.5)

    ax = plt.gca()
    box = ax.get_position()
    plt.show()
    # %%
    utils.savefig(fig, "icassp2023-dynamic-time", format="pgf", pgf_font="serif")

    # %%
    # Plot
    # mavg = 200
    # styles = ["-+", "-x", "-<", "->", "-v", "-s", "-o", "k-"]
    # fig = plt.figure(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.4))
    # # plt.title("Title")
    # # plt.xlabel("Time [frames]")
    # plt.ylabel(r"$\|\mathbf{h}\|$ [1]")
    # plt.vlines(
    #     [5000, 10000],
    #     0.75,
    #     1.25,
    #     colors=["k", "k"],
    #     linestyles=["dashed", "dashed"],
    #     linewidth=[0.75, 0.75],
    # )
    # (line,) = plt.plot(
    #     np.sqrt(data.mean().T["davgad", 0.0, 1][frames : 2 * frames].to_numpy()),
    #     "-",
    # )
    # # #########
    # # plt.legend(ncol=2, prop={"size": 7}, columnspacing=0.5)
    # plt.grid()
    # plt.ylim(0.75, 1.25)
    # plt.xlim(0, 15000)
    # # plt.tight_layout(pad=0.5)
    # ax = plt.gca()
    # ax.set_xticklabels([])
    # # ax.set_position(box.shrunk(1.0, relative_height))
    # ax.set_position(box)
    # plt.show()

    # %%
    # utils.savefig(fig, "icassp2023-dynamic-norm", format="pgf", pgf_font="serif")

    # %%
    # Plot
    true_norms = [[2.2, 0.5, 1.2], [2.2, 1.0, 1.2], [2.2, 0.5, 2.0]]
    true_normed_norms = np.asarray(true_norms)
    for n in range(true_normed_norms.shape[0]):
        true_normed_norms[n, :] = np.square(true_normed_norms[n, :]) / np.sum(
            np.square(true_normed_norms[n, :])
        )
    mavg = 200
    styles = ["-+", "-x", "-<", "->", "-v", "-s", "-o", "k-"]
    fig = plt.figure(figsize=utils.set_size(textwidth, 1.0, (1, 1), 0.5))
    plt.ylabel(r"$\|\hat{\mathbf{h}}_i\|, \|\hat{\mathbf{h}}\|$")
    (line,) = plt.plot(
        np.sqrt(data.mean().T["davgad", 0.0, 1][2 * frames : 3 * frames].to_numpy()),
        "-o",
        label=r"$\|\hat{\mathbf{h}}_1\|$",
        markersize=4,
        markevery=[2000, 7000, 12000],
        linewidth=linewidth,
        markerfacecolor="none",
    )
    (line,) = plt.plot(
        np.sqrt(data.mean().T["davgad", 0.0, 1][3 * frames : 4 * frames].to_numpy()),
        "-s",
        label=r"$\|\hat{\mathbf{h}}_2\|$",
        markersize=4,
        markevery=[2500, 7500, 12500],
        linewidth=linewidth,
        markerfacecolor="none",
    )
    (line,) = plt.plot(
        np.sqrt(data.mean().T["davgad", 0.0, 1][4 * frames : 5 * frames].to_numpy()),
        "-v",
        label=r"$\|\hat{\mathbf{h}}_3\|$",
        markersize=4,
        markevery=[3000, 8000, 13000],
        linewidth=linewidth,
        markerfacecolor="none",
    )
    (line,) = plt.plot(
        np.sqrt(data.mean().T["davgad", 0.0, 1][frames : 2 * frames].to_numpy()),
        "-",
        label=r"$\|\hat{\mathbf{h}}\|$",
        markerfacecolor="none",
        linewidth=linewidth,
    )
    plt.plot(
        [0, 5000, 5000, 10000, 10000, 15000],
        np.repeat(np.sqrt(true_normed_norms), 2, axis=0),
        "-.",
        linewidth=0.8,
        label=["true", "_true", "_true"],
        color="k",
    )
    plt.plot(
        [0, 15000],
        np.repeat(1, 2, axis=0),
        "-.",
        linewidth=0.8,
        color="k",
    )
    # #########
    plt.legend(
        ncol=5,
        prop={"size": 7},
        columnspacing=0.5,
        handlelength=1,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.74),
    )
    plt.grid()
    plt.ylim(0, 1.5)
    plt.xlim(0, 15000)
    # plt.tight_layout(pad=0.5)
    ax = plt.gca()
    ax.set_xticklabels([])
    # ax.set_position(box.shrunk(1.0, relative_height))
    ax.set_position(box)
    plt.show()
    # %%
    utils.savefig(fig, "icassp2023-dynamic-norms", format="pgf", pgf_font="serif")

# %%
