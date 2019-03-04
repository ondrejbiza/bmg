import matplotlib.pyplot as plt
from ..data import Data1D
from ..bmg import UBMG


data = Data1D(100)
data.show()

model = UBMG(len(data.clusters), 1.0)

elbos = model.fit(data.points, show=True, clusters=data.clusters)

plt.plot(list(range(1, len(elbos) + 1)), elbos)
plt.xlabel("step")
plt.ylabel("ELBO")
plt.show()


points = model.sample(300)
plt.title("sampled points")
plt.scatter(points, [0] * len(points))
plt.yticks([])
plt.xlabel("value")
plt.show()
