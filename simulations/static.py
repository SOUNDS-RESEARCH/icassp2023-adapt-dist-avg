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
    stepsize = 0.5
    eta = 0.98
    N_sens = nw.N
    N_s = 100000

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
    cfg = simulation.SimConfig(
        id="icassp2023-static",
        runs=30,
        seed=573438,
        variables=[
            {"alg": ["opt"], "SNR": [10], "gamma": [1.0], "iters": [1]},
            {"alg": ["davgad"], "SNR": [10], "gamma": [1.0], "iters": [1]},
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
                    0.022,
                    0.024,
                    0.026,
                    0.028,
                    0.03,
                    0.032,
                    0.034,
                    0.036,
                    0.038,
                    0.04,
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
    data = result.df.groupby(["alg", "gamma", "iters"])

    # %%
    styles = ["-v", "-+", "-x", "-s", "-o", "k-"]
    fig = plt.figure(figsize=utils.set_size(245, 1.0, (1, 1)))
    # plt.title(f"SNR={cfg.variables['SNR'][0]}dB")
    plt.xlabel(r"Mixing factor $\gamma$ [1]")
    plt.ylabel("Avg. NPM [dB]")

    plt.plot(
        cfg.variable_values[2]["gamma"],
        20
        * np.log10(data.median().T.tail(200).mean()["opt", 1.0, 1])
        * np.ones_like(cfg.variable_values[2]["gamma"]),
        styles.pop(),
        label=f"optimal",
        markersize=4,
        markevery=(1, 2),
    )
    plt.plot(
        cfg.variable_values[2]["gamma"],
        20
        * np.log10(data.median().T.tail(200).mean()["davgad", 1.0, 1])
        * np.ones_like(cfg.variable_values[2]["gamma"]),
        styles.pop(),
        label=rf"adaptive $\gamma_i,\,K=1$",
        markersize=4,
        markevery=(2, 2),
    )
    plt.plot(
        cfg.variable_values[2]["gamma"],
        20 * np.log10(data.median().T.tail(200).mean()["davg", :, 1]),
        styles.pop(),
        label=rf"fixed $\gamma_i,\,K=1$",
        markersize=4,
        markevery=(1, 2),
    )
    plt.plot(
        cfg.variable_values[2]["gamma"],
        20 * np.log10(data.median().T.tail(200).mean()["davg", :, 2]),
        styles.pop(),
        label=rf"fixed $\gamma_i,\,K=2$",
        markersize=4,
        markevery=(2, 2),
    )
    plt.plot(
        cfg.variable_values[2]["gamma"],
        20 * np.log10(data.median().T.tail(200).mean()["davg", :, 10]),
        styles.pop(),
        label=rf"fixed $\gamma_i,\,K=10$",
        markersize=4,
        markevery=(1, 2),
    )

    plt.grid()
    plt.ylim(-40, -20)
    plt.xlim(0, 0.04)
    plt.legend(ncol=2, prop={"size": 7}, columnspacing=0.5)
    plt.tight_layout(pad=0.5)
    plt.show()

    # %%
    utils.savefig(fig, "icassp2023-static", format="pgf", pgf_font="serif")

    # %%
    # Plot
    mavg = 200
    styles = ["-+", "-x", "-<", "->", "-v", "-s", "-o", "k-"]
    fig = plt.figure(figsize=utils.set_size(245, 1.0, (1, 1)))
    # plt.title("Title")
    plt.xlabel("Time [frames]")
    plt.ylabel("NPM [dB]")
    # #########
    (line,) = plt.plot(
        20 * np.log10(data.median().T["opt", 1.0, 1]),
        "k-",
        label=f"_optimal",
        markersize=4,
        markevery=(1, 500),
        alpha=0.25,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["opt", 1.0, 1], np.ones(mavg) / mavg, mode="valid"
            )
        ),
        "-",
        label=f"optimal",
        markersize=4,
        markevery=(1, 500),
        color=line.get_color(),
    )
    # #########
    (line,) = plt.plot(
        20 * np.log10(data.median().T["davgad", 1.0, 1]),
        "-",
        label=rf"_adaptive $\gamma_i,\,K=1$",
        markersize=4,
        markevery=(20, 500),
        alpha=0.25,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["davgad", 1.0, 1], np.ones(mavg) / mavg, mode="valid"
            )
        ),
        "-o",
        label=rf"adaptive $\gamma_i,\,K=1$",
        markersize=4,
        markevery=(20, 500),
        color=line.get_color(),
    )
    # #########
    (line,) = plt.plot(
        20 * np.log10(data.median().T["davg", 0.01, 1]),
        "-",
        label=rf"_$\gamma_i = 0.01,\,K=1$",
        markersize=4,
        markevery=(40, 500),
        alpha=0.25,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["davg", 0.01, 1], np.ones(mavg) / mavg, mode="valid"
            )
        ),
        "-s",
        label=rf"$\gamma_i = 0.01,\,K=1$",
        markersize=4,
        markevery=(40, 500),
        color=line.get_color(),
    )
    # #########
    (line,) = plt.plot(
        20 * np.log10(data.median().T["davg", 0.01, 10]),
        "-",
        label=rf"_$\gamma_i = 0.01,\,K=10$",
        markersize=4,
        markevery=(60, 500),
        alpha=0.25,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["davg", 0.01, 10], np.ones(mavg) / mavg, mode="valid"
            )
        ),
        "-x",
        label=rf"$\gamma_i = 0.01,\,K=10$",
        markersize=4,
        markevery=(60, 500),
        color=line.get_color(),
    )
    # #########
    (line,) = plt.plot(
        20 * np.log10(data.median().T["davg", 0.04, 1]),
        "-",
        label=rf"_$\gamma_i = 0.04,\,K=1$",
        markersize=4,
        markevery=(80, 500),
        alpha=0.25,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["davg", 0.04, 1], np.ones(mavg) / mavg, mode="valid"
            )
        ),
        "-+",
        label=rf"$\gamma_i = 0.04,\,K=1$",
        markersize=4,
        markevery=(80, 500),
        color=line.get_color(),
    )
    # #########
    (line,) = plt.plot(
        20 * np.log10(data.median().T["davg", 0.04, 10]),
        "-",
        label=rf"_$\gamma_i = 0.04,\,K=10$",
        markersize=4,
        markevery=(100, 500),
        alpha=0.25,
    )
    plt.plot(
        20
        * np.log10(
            np.convolve(
                data.median().T["davg", 0.04, 10], np.ones(mavg) / mavg, mode="valid"
            )
        ),
        "-v",
        label=rf"$\gamma_i = 0.04,\,K=10$",
        markersize=4,
        markevery=(100, 500),
        color=line.get_color(),
    )
    # #########
    plt.legend(ncol=2, prop={"size": 7}, columnspacing=0.5)
    plt.grid()
    plt.tight_layout(pad=0.5)
    plt.show()
    # %%
    utils.savefig(fig, "icassp2023-static-time", format="pgf", pgf_font="serif")

# %%
