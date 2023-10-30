import matplotlib.pyplot as plt
import numpy as np

data = []
avg = []
ten = np.zeros(10)
with open('scores.txt', 'r') as file:
    for line in file:
        data.append(float(line.strip()))
for i in range(len(data)):
    ten[i % 10] = data[i]
    avg.append(np.mean(ten))


plt.plot(avg)
plt.xlabel('episodes')
plt.ylabel('score')

# plt.grid(True)
plt.show()
