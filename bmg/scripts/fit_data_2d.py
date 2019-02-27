import matplotlib.pyplot as plt
from ..data import Data2D
from ..bmg import BMG


data = Data2D(100)
model = BMG(len(data.clusters), 0.1)

model.fit(data.points)

for i in range(10000):

    #print(model.phis)

    model.means_assignment(data.points)
    model.mixture_assignments(data.points)

    #print(model.means)
    #print(model.vars)

plt.scatter(model.means[:, 0], model.means[:, 1])
data.show()
