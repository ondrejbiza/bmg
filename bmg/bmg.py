import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


class UBMG:

    def __init__(self, num_components, variance):

        self.num_components = num_components
        self.variance = variance

        self.num_samples, self.phis, self.means, self.vars = [None] * 4

    def fit(self, data, max_steps=1000, tolerance=1e-6, show=False, clusters=None):

        self.num_samples = data.shape[0]

        self.phis = np.random.dirichlet([1.0] * self.num_components)
        self.means = np.random.uniform(-1, 1, size=self.num_components)
        self.vars = np.random.uniform(0, 0.1, size=self.num_components)

        elbos = [self.elbo(data)]

        for step_idx in range(max_steps):

            if show:
                self.show(clusters)

            self.mixture_assignments(data)
            self.means_assignment(data)

            elbos.append(self.elbo(data))

            diff = np.abs(elbos[-1] - elbos[-2])

            if diff <= tolerance:
                break

        if show:
            self.show(clusters)

        return elbos

    def mixture_assignments(self, data):

        term1 = np.outer(data, self.means)
        term2 = (0.5 * np.square(self.means) + 0.5 * self.vars)[np.newaxis, :]

        assignment = term1 - term2
        assignment = np.exp(assignment)
        assignment = assignment / assignment.sum(axis=1)[:, np.newaxis]

        self.phis = assignment

    def means_assignment(self, data):

        term1 = self.phis * data[:, np.newaxis]
        term1 = term1.sum(axis=0)

        term2 = 1 / self.variance + self.phis.sum(axis=0)

        self.means = term1 / term2
        self.vars = 1 / term2

    def elbo(self, data):

        # prior on the component mean
        t1 = - (1 / 2) * np.log(2 * np.pi * self.variance) - \
             (1 / (2 * self.variance)) * (np.square(self.means) + self.vars)
        t1 = t1.sum()

        # prior on the assignments
        t2 = data.shape[0] * np.log(1 / self.num_components)

        # likelihood of the data
        t3a = - self.phis * (1 / 2) * np.log(2 * np.pi)
        t3b = - self.phis * (np.square(data) * (1 / 2))[:, np.newaxis]
        t3c = self.phis * (data[:, np.newaxis] * self.means[np.newaxis, :])
        t3d = - self.phis * ((np.square(self.means) + self.vars) * (1 / 2))[np.newaxis, :]

        t3 = t3a + t3b + t3c + t3d
        t3 = np.sum(t3)

        # entropy of the assignment
        t4 = self.phis * np.log(self.phis)
        t4[np.isnan(t4)] = 0.0     # for 0 * np.log(0), shouldn't happen anyways
        t4 = np.sum(t4)

        # entropy of the means
        t5 = (1 / 2) * np.log(2 * np.pi * self.vars) + (1 / 2)
        t5 = np.sum(t5)

        return t1 + t2 + t3 + t4 + t5

    def show(self, clusters):

        colors = sns.color_palette()

        for idx, cluster in enumerate(clusters):

            x = np.linspace(-10, 10, 1000)
            y = norm.pdf(x, loc=self.means[idx], scale=1.0)

            plt.plot(x, y, c=colors[idx])
            sns.distplot(cluster, hist=True, norm_hist=True, kde=False, color=colors[idx + len(clusters)])

        plt.xlabel("value")
        plt.ylabel("probability")
        plt.show()

    def sample(self, num_points):

        categories = np.mean(self.phis, axis=0)

        assignments = np.random.choice(list(range(self.num_components)), size=num_points, replace=True, p=categories)

        points = np.empty(num_points, dtype=np.float32)

        for idx in range(self.num_components):

            mask = [assignments == idx]
            count = np.sum(mask)
            points[mask] = np.random.normal(self.means[idx], scale=np.sqrt(self.variance), size=count)

        return points
