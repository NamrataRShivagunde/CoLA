import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt


def certify(x, pA=0.9, sigma=0.003):
    return norm.cdf(norm.ppf(pA) - x / sigma)


x = np.arange(0, 15.0, 0.25)
x = 2**x
x /= 1000
print(x)

plt.plot(x, certify(x, sigma=0.003), label="0.003")
plt.plot(x, certify(x, sigma=0.03), label="0.03")
plt.plot(x, certify(x, sigma=0.3), label="0.3")
plt.plot(x, certify(x, sigma=3), label="3")
# plt.plot(x, certify(x, 0.99999), label="0.99999")

plt.xscale("log")
plt.xlabel(r"$\|\mathbf{\theta}_{sft} - \mathbf{\theta}_0\|_2$")
plt.ylabel(r"Lower bound on $J_{\mathcal{D}}(f_{\mathbf{\theta}_{sft} + \mathbf{\epsilon}})$")
plt.legend()
plt.savefig("./qwen-lower-bound-sigma.pdf", bbox_inches="tight")
plt.show()
