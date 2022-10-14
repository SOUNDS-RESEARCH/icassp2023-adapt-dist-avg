# %%
import numpy as np
import python_utils.utils as utils

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
M = np.arange(10, 5000)

# %%
N_bar = [5, 10]

# %%
R = [1, 50]

# %%
# fig = plt.figure(figsize=(6, 4))
# # plt.title("Title")
# plt.xlabel("Network size M [1]")
# plt.ylabel(r"$\log \mathcal{O}(.)$ [1]")
# plt.plot(M,2*M**2, "k--", label="full direct")
# plt.plot(M,M+M**2, "-.", label="broadcast")
# for n_bar in N_bar:
#     for r in R:
#         plt.plot(M,M*n_bar*r + M*n_bar*r, label=r"$\bar{N}=$%d,$R=$%d"%(n_bar, r))
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# %%
styles = ["-<", "->", "-v", "-s", "-o"]
fig = plt.figure(figsize=utils.set_size(245, 1.0, (1, 1)))
# plt.title("Title")
plt.xlabel("Network size M [1]")
plt.ylabel(r"$\mathcal{O}(.)$ [1]")
plt.plot(M, 2 * M**2, "k-", label="full direct")
plt.plot(M, M + M**2, "--", label="broadcast")
ss = 0
for n_bar in N_bar:
    for r in R:
        plt.plot(
            M,
            M * n_bar * r + M * n_bar * r,
            styles.pop(),
            label=r"$\bar{N}=$%d,$R=$%d" % (n_bar, r),
            markevery=(ss, 500),
            markersize=3,
        )
        ss += 50
plt.legend(ncol=2, prop={"size": 7})
plt.grid()
plt.yscale("log")
plt.xlim(0, 5000)
plt.tight_layout()
plt.show()
# %%
utils.savefig(fig, "transcost", format="pgf", pgf_font="serif")

# %%
styles = ["-<", "-+", "-x", "-s", "-o"]
fig = plt.figure(figsize=utils.set_size(245, 0.95, (1, 1)))
# plt.title("Title")
plt.xlabel("Network size M [1]")
plt.ylabel(r"$\mathcal{O}(.)$ [1]")
plt.plot(M, 2 * M, "k-", label="full direct")
plt.plot(M, 1 + M, "--", label="broadcast")
ss = 0
for n_bar in N_bar:
    for r in R:
        plt.plot(
            M,
            np.ones_like(M) * n_bar * r + n_bar * r,
            styles.pop(),
            label=r"$\bar{N}=$%d,$R=$%d" % (n_bar, r),
            markevery=(ss, 500),
            markersize=4,
            markerfacecolor="none",
        )
        ss += 50
plt.legend(ncol=2, prop={"size": 7}, loc="center left", bbox_to_anchor=(0.1, 0.35))
plt.grid()
plt.yscale("log")
plt.xlim(0, 5000)
plt.tight_layout(pad=0.5)
plt.show()
# %%
utils.savefig(fig, "transcostnode", format="pgf", pgf_font="serif")
# %%
