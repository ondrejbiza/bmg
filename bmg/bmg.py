import numpy as np


class UBMG:
    # for multi, try this: https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/IDAPISlides17_18.pdf

    def __init__(self, num_components, component_means_var):

        self.num_components = num_components
        self.component_means_var = component_means_var

    def fit(self, data):

        self.num_samples = data.shape[0]

        self.phis = np.random.dirichlet([1.0] * self.num_components)
        self.means = np.random.uniform(-1, 1, size=self.num_components)
        self.vars = np.random.uniform(0, 0.1, size=self.num_components)

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

        term2 = 1 / self.component_means_var + self.phis.sum(axis=0)

        self.means = term1 / term2
        self.vars = 1 / term2

    def elbo(self):

        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        t5 = 0