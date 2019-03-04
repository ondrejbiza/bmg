import matplotlib.pyplot as plt
from ..data import Data1D
from ..bmg import UBMG


# generate data
data = Data1D(100)
data.show()

# setup model
model = UBMG(len(data.clusters), 1.0)

# fit model
elbos = model.fit(data.points, show=True, clusters=data.clusters)

# plot ELBO
plt.title("Evidence Lower Bound for GMM")
plt.plot(list(range(1, len(elbos) + 1)), elbos)
plt.xlabel("step")
plt.ylabel("ELBO")
plt.show()

# plot samples
points = model.sample(300)
plt.title("sampled points")
plt.scatter(points, [0] * len(points))
plt.yticks([])
plt.xlabel("value")
plt.show()
