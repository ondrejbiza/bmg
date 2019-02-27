import numpy as np


class BMG:

    def __init__(self, num_components, component_means_var):

        self.num_components = num_components
        self.component_means_var = component_means_var

    def fit(self, data):

        self.num_samples = data.shape[0]

        self.phis = np.random.dirichlet(np.ones(self.num_components, dtype=np.float32), size=self.num_samples)

    def mixture_assignments(self, data):

        assignment = data * self.means[:, np.newaxis, :] - (np.square(self.means) + self.vars)[:, np.newaxis, :] / 2
        assignment = assignment.sum(axis=2)
        assignment = assignment.transpose()
        #assignment = np.exp(assignment)
        assignment = assignment / assignment.sum(axis=1)[:, np.newaxis]

        self.phis = assignment

    def means_assignment(self, data):

        term1 = data[:, np.newaxis, :] * self.phis[:, :, np.newaxis]
        term1 = term1.sum(axis=0)

        term2 = 1 / self.component_means_var + self.phis.sum(axis=0)

        self.means = term1 / term2[:, np.newaxis]

        self.vars = 1 / term2
        self.vars = np.stack([self.vars for _ in range(self.means.shape[1])], axis=1)
