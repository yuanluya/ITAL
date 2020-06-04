import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb

R = 3
w = np.random.uniform(low = -10, high = 10, size = 2)
w /= np.sqrt(np.sum(np.square(w)))
num_samples = 100000
Xs = []
for i in range(num_samples):
	while True:
		x = np.random.uniform(low = -1 * R, high = R, size = 2)
		if np.sqrt(np.sum(x[0] ** 2 + x[1] ** 2)) <= R:
		# if abs(np.square(x[0] + x[1]) - R) < 1e-6:
			break
	Xs.append(copy.deepcopy(x))
Xs = np.array(Xs)
inner_products = np.sum(np.expand_dims(x, 0) * Xs, axis = 1, keepdims = True)
newXs1 = inner_products * Xs
newXs2 = -1 * np.exp(-1 * inner_products) / (1 + np.exp(-1 * inner_products)) * Xs

# # plt.scatter(newXs1[:, 0], newXs1[:, 1])
# # plt.scatter(newXs2[:, 0], newXs2[:, 1])
# plt.scatter(Xs[:, 0], Xs[:, 1])
# # plt.scatter(w[0], w[1])
# plt.title("%f, %f" % (w[0], w[1]))
# plt.show()
newXs2 = -1 * Xs / (1 + np.exp(inner_products))
# plt.scatter(newXs1[:, 0], newXs1[:, 1])
plt.scatter(newXs2[:, 0], newXs2[:, 1])
# plt.scatter(Xs[:, 0], Xs[:, 1])
plt.scatter(w[0], w[1])
plt.title("%f, %f" % (w[0], w[1]))
plt.show()
