import numpy as np
import matplotlib.pyplot as plt


class ExplorationNoise(object):
    def __init__(self, mu, sigma=0.5, theta=0.2, dt=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = 0.0
        self.decay_count = 0

    def ou_noise(self, sigma_decay=False):
        if sigma_decay:
            if self.sigma > 0.15:
                self.decay_count += self.dt / 10000
                self.sigma = self.sigma * np.exp(-self.decay_count)
        dW = np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        x = self.x_prev + (self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * dW)
        self.x_prev = x
        return x


noise = ExplorationNoise(mu=np.zeros(1))
n = []
for i in range(10000):
    n.append(noise.ou_noise(sigma_decay=True))

plt.plot(n)
plt.grid()
plt.show()
