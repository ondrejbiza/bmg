import matplotlib.pyplot as plt
from ..data import Data1D
from ..bmg import UBMG


data = Data1D(100)
model = UBMG(len(data.clusters), 1.0)

model.fit(data.points)

for i in range(100):

    plt.scatter(model.means, [len(data.clusters)] * len(model.means))
    data.show()

    model.mixture_assignments(data.points)
    model.means_assignment(data.points)
