import argparse
import numpy as np
import matplotlib.pyplot as plt
from ..data import Data2DDiag
from ..gmm_tf import GMM_TF


def main(args):

    # parameters
    num_components = args.num_components
    num_steps = args.num_steps
    learning_rate = args.learning_rate
    prior_entropy_weight = args.prior_entropy_weight

    # generate data
    data = Data2DDiag(1000)
    data.show()

    # setup model
    model = GMM_TF(num_components, 2)

    # fit model
    lls = model.fit(
        data.points, num_steps=num_steps, learning_rate=learning_rate, prior_entropy_weight=prior_entropy_weight
    )

    # plot ELBO
    plt.title("Log likelihood for GMM")
    plt.plot(list(range(1, len(lls) + 1)), lls)
    plt.xlabel("step")
    plt.ylabel("log likelihood")
    plt.show()

    # plot samples
    points, classes = model.sample(1000 * len(data.clusters))
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 1, 1)
    plt.title("sampled points")

    for i in range(num_components):
        mask = classes == i
        if np.sum(mask):
            plt.scatter(points[mask][:, 0], points[mask][:, 1], label="c{:d}".format(i))
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(data.points[:, 0], data.points[:, 1])
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num-components", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--prior-entropy-weight", type=float, default=0.0)

    parsed = parser.parse_args()
    main(parsed)
