import matplotlib.pyplot as plt

data = []
with open('scores.txt', 'r') as file:
    for line in file:
        data.append(float(line.strip()))

plt.plot(data)
plt.xlabel('episodes')
plt.ylabel('score')

plt.grid(True)
plt.show()
