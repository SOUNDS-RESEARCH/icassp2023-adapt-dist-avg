import numpy as np


class NodeProcessor:
    def __init__(self, id, filter_len, rho) -> None:
        # identifier for node
        self.id = id
        # iterator
        self.k = 0
        # local solver type
        self.type = type
        # definition of signal connections
        self.group = None
        # number of FIR filter taps
        self.L = filter_len
        # step size / penalty parameter
        self.rho = rho
        # step size / penalty parameter
        self.mu = 1
        self.eta = 0.98
        # number of "channels"
        self.N = None
        # signal block
        self.block = None
        # local primal
        self.x = None
        # local dual
        self.y = None
        # local dual est old
        self.z_l = None
        # estimated
        self.R_xp_ = None
        # buffer
        self.R_buffer = None

        self.setBufferSize(1)
        pass

    def setLocalSize(self, N) -> None:
        self.N = N
        # global indices
        self.global_indices = None
        self.reset()
        pass

    def setBufferSize(self, buffer_size) -> None:
        self.buffer_size = buffer_size
        self.buffer_ind = 0
        # self.reset()
        pass

    def reset(self) -> None:
        # signal block
        self.block = np.zeros(shape=(self.L, self.N), dtype=np.complex128)
        # local primal
        self.x = np.zeros(shape=(self.L * self.N, 1), dtype=np.complex128)
        # self.x[0::self.L] = 1
        # local dual
        self.y = np.zeros(shape=(self.L * self.N, 1), dtype=np.complex128)
        # local copy of global primal
        self.z_l = np.ones(shape=(self.L * self.N, 1), dtype=np.complex128) / np.sqrt(
            self.L
        )

        F_LxL = self.DFT_matrix(self.L)
        F_2Lx2L = self.DFT_matrix(2 * self.L)

        W_01_Lx2L = np.concatenate([np.zeros(shape=(self.L, self.L)), np.eye(self.L)]).T
        W_10_2LxL = np.concatenate([np.eye(self.L), np.zeros(shape=(self.L, self.L))])

        self.W_01_Lx2L_fq = F_LxL @ W_01_Lx2L @ np.linalg.inv(F_2Lx2L)
        self.W_10_2LxL_fq = F_2Lx2L @ W_10_2LxL @ np.linalg.inv(F_LxL)

        W_10_Lx2L = np.concatenate([np.eye(self.L), np.zeros(shape=(self.L, self.L))]).T
        W_01_2LxL = np.concatenate([np.zeros(shape=(self.L, self.L)), np.eye(self.L)])

        self.W_10_Lx2L_fq = F_LxL @ W_10_Lx2L @ np.linalg.inv(F_2Lx2L)
        self.W_01_2LxL_fq = F_2Lx2L @ W_01_2LxL @ np.linalg.inv(F_LxL)

        self.W_R = self.W_01_Lx2L_fq.conj().T @ self.W_01_Lx2L_fq

        # print(self.W_01_Lx2L_fq.shape)
        # print(self.W_10_2LxL_fq.shape)
        # print(self.W_01_Lx2L_fq)
        # print(self.W_10_2LxL_fq)
        # print(self.W_10_Lx2L_fq)
        # print(self.W_01_2LxL_fq)

        # self.buffer_size = 1
        # self.R_buffer = np.zeros(
        #     shape=(self.buffer_size, self.N, 2 * self.L), dtype=np.complex128
        # )
        # self.buffer_ind = 0
        # self.buffer_filled = False
        self.first = True
        self.R_norm = []
        self.V_norm = []
        self.x_norm = []

        pass

    def setRho(self, rho, mu, eta) -> None:
        self.rho = rho
        self.mu = mu
        self.eta = eta
        pass

    def receiveSignal(self, signal) -> None:
        self.block = np.fft.fft(signal, axis=0)
        # self.R_buffer[self.buffer_ind, :, :] = self.block.T
        # self.buffer_ind += 1
        # if self.buffer_ind >= self.buffer_size:
        #     self.buffer_ind = 0
        #     self.buffer_filled = True
        # pass

    def step(self) -> None:
        # if not self.buffer_filled:
        #     return
        self.solveLocalLS()
        pass

    def solveLocalLS(self) -> None:
        R = self.construct_Rxp()  # construct matrix R_x+
        self.R_xp_ = R if self.first else self.eta * self.R_xp_ + (1 - self.eta) * R
        y = self.R_xp_ @ self.x + self.y + self.rho * (self.x - self.z_l)
        V = 1 / (np.diag(self.R_xp_).reshape(self.N * self.L, 1) + self.rho)
        # self.V = V if self.first else self.eta * self.V + (1 - self.eta) * V
        self.first = False
        self.x = self.x - self.mu * V * y
        pass

    def construct_Rxp(self):
        R_xp = np.zeros(shape=(self.L * self.N, self.L * self.N), dtype=np.complex128)
        for i in range(self.N):
            for j in range(self.N):
                R_xx = self.compute_Rxx(i, j)
                if i != j:
                    # R_xp[j, i] = -R_xx
                    R_xp[
                        j * self.L : j * self.L + self.L,
                        i * self.L : i * self.L + self.L,
                    ] = -R_xx
                else:
                    for n in range(self.N):
                        if i != n:
                            # R_xp[n, n] += R_xx
                            R_xp[
                                n * self.L : n * self.L + self.L,
                                n * self.L : n * self.L + self.L,
                            ] += R_xx

        return R_xp

    def compute_Rxx(self, i, j):
        # D_xi = np.diag(np.mean(self.R_buffer[:, i, :], axis=0))
        # D_xj = np.diag(np.mean(self.R_buffer[:, j, :], axis=0))
        D_xi = np.diag(self.block[:, i])
        D_xj = np.diag(self.block[:, j])

        R_xx = self.W_10_Lx2L_fq @ D_xi.conj().T @ self.W_R @ D_xj @ self.W_10_2LxL_fq

        return R_xx

    def dualUpdate(self) -> None:
        # if not self.buffer_filled:
        #     return
        self.y = self.y + self.rho * (self.x - self.z_l)
        pass

    def DFT_matrix(self, N):
        i, j = np.meshgrid(np.arange(N), np.arange(N))
        omega = np.exp(-2 * np.pi * 1j / N)
        W = np.power(omega, i * j) / np.sqrt(N)
        return W


class Network:
    def __init__(self, L) -> None:
        # number of FIR filter taps
        self.L = L
        # number of nodes in network
        self.N = 0
        # central processor for gloabl update
        self.central_processor = None
        # node objecs
        self.nodes = {}
        # connections between nodes
        self.connections = {}
        # node index
        self.node_index = {}
        # global primal variable
        self.z = None
        # network matrix
        self.A = None
        self.A_ = None
        # global update weights
        self.g = None
        self.counter = 0
        self.global_ds = None  # global update only done each K frames
        pass

    def reset(self) -> None:
        # self.z = np.zeros(shape=(self.N*self.L, 1))
        for node_key, node in self.nodes.items():
            node.reset()
        self.zs = np.array([]).reshape(0, self.L * self.N)
        self.counter = 0
        pass

    def addNode(self, id, rho) -> int:
        self.nodes[id] = NodeProcessor(id, self.L, rho)
        self.N = len(self.nodes)
        self.z = np.zeros(shape=(self.N * self.L, 1))
        self.generateNetworkData()
        pass

    def removeNode(self, id) -> None:
        del self.nodes[id]
        self.N = len(self.nodes)
        self.z = np.zeros(shape=(self.N * self.L, 1))
        self.generateNetworkData()
        pass

    def setConnection(self, node_id, connections) -> None:
        self.connections[node_id] = connections
        self.generateNetworkData()
        pass

    def generateNetworkData(self) -> None:
        self.A = np.zeros(shape=(self.N, self.N))
        self.A_ = np.diag(np.repeat(1, self.N))
        self.node_index = {}
        for i, node_key in enumerate(self.nodes):
            self.node_index[node_key] = i

        for i, node_key in enumerate(self.nodes):
            if node_key in self.connections:
                for connection in self.connections[node_key]:
                    j = self.node_index[connection]
                    self.A[i, j] = 1
                    self.A_[i, j] = 1

        for node_key in self.node_index:
            i = self.node_index[node_key]
            self.nodes[node_key].setLocalSize(np.sum(self.A_[:, i]))
            self.nodes[node_key].global_indices = (
                np.tile(np.arange(self.L), self.nodes[node_key].N)
                + np.repeat(np.where(self.A_[:, i]), self.L) * self.L
            )

        self.g = np.repeat(1 / np.sum(self.A_, axis=1), self.L).reshape(
            self.N * self.L, 1
        )
        pass

    def step(self, signal) -> None:
        self.broadcastSignals(signal)
        self.localPrimalUpdate()
        if self.counter == 0:
            self.globalUpdate()
            self.broadcastGlobalVariable()

        self.counter += 1
        if self.counter >= self.global_ds:
            self.counter -= self.global_ds
        self.localDualUpdate()
        pass

    def broadcastSignals(self, signal) -> None:
        for node_key, node in self.nodes.items():
            i = self.node_index[node_key]
            local_signal = signal[:, self.A_[:, i] == 1]
            node.receiveSignal(local_signal)
        pass

    def localPrimalUpdate(self) -> None:
        for node_key, node in self.nodes.items():
            node.step()
        pass

    def globalUpdate(self) -> None:
        self.z = np.zeros(shape=(self.N * self.L, 1), dtype=np.complex128)
        x_avg = np.zeros(shape=(self.N * self.L, 1), dtype=np.complex128)
        y_avg = np.zeros(shape=(self.N * self.L, 1), dtype=np.complex128)
        for node_key, node in self.nodes.items():
            # if not node.buffer_filled:
            #     return
            x_avg[node.global_indices] += node.x * self.g[node.global_indices]
            y_avg[node.global_indices] += node.y * self.g[node.global_indices]
        pp = node.rho * x_avg + y_avg  # WHAT DO WITH RHO IF NOT SAME FOR ALL
        self.z = pp / (np.linalg.norm(pp))
        pass

    def broadcastGlobalVariable(self) -> None:
        for node_key, node in self.nodes.items():
            # if not node.buffer_filled:
            #     return
            node.z_l = self.z[node.global_indices]
        pass

    def localDualUpdate(self) -> None:
        for node_key, node in self.nodes.items():
            node.dualUpdate()
        pass

    def setBufferSize(self, buffer_size) -> None:
        for node_key, node in self.nodes.items():
            node.setBufferSize(buffer_size)
        pass

    def setRho(self, rho, mu, eta, global_ds) -> None:
        self.global_ds = global_ds
        for node_key, node in self.nodes.items():
            node.setRho(rho, mu, eta)
        pass
