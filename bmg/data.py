import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")


class Data1D:

    MEANS = np.array([-7.0, 0.0, 7.0], dtype=np.float32)
    VARS = np.array([1.0, 1.0, 1.0])

    def __init__(self, points_per_cluster=1000):

        self.clusters = []

        for i in range(len(self.MEANS)):

            cluster = np.random.normal(self.MEANS[i], scale=np.sqrt(self.VARS[i]), size=points_per_cluster)

            self.clusters.append(cluster)

        self.points = np.concatenate(self.clusters, axis=0)

    def show(self):

        for idx, cluster in enumerate(self.clusters):

            plt.scatter(cluster, [idx] * len(cluster))

        plt.axis("off")
        plt.xlim(-10.0, 10.0)
        plt.ylim(-0.5, len(self.clusters) + 1)

        plt.show()


class Data2D:

    MEANS = np.array([
        [-1.0, 1.0],
        [-0.2, 0.8],
        [1.0, 1.0],
        [-1.0, -0.6],
        [0.2, -1.0]
    ], dtype=np.float32)

    COV = np.array([
        [
            [0.05, 0.03],
            [0.03, 0.05]
        ],
        [
            [0.05, -0.01],
            [-0.01, 0.1]
        ],
        [
            [0.05, -0.035],
            [-0.035, 0.05]
        ],
        [
            [0.05, 0.0],
            [0.0, 0.1]
        ],
        [
            [0.05, -0.02],
            [-0.02, 0.05]
        ]
    ], dtype=np.float32)

    def __init__(self, points_per_cluster=1000):

        self.clusters = []

        for i in range(len(self.MEANS)):

            cluster = np.random.multivariate_normal(self.MEANS[i], cov=self.COV[i], size=points_per_cluster)

            self.clusters.append(cluster)

        self.points = np.concatenate(self.clusters, axis=0)


    def show(self):

        for cluster in self.clusters:

            plt.scatter(cluster[:, 0], cluster[:, 1])

        plt.axhline(0, linewidth=0.8, color="black")
        plt.axvline(0, linewidth=0.8, color="black")
        plt.axis("off")
        plt.xlim(-2.0, 2.0)
        plt.ylim(-2.0, 2.0)

        plt.show()
