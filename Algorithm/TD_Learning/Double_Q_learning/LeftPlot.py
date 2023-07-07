import matplotlib.pyplot as plt
import numpy as np


def show_figure(prob_E_A_left):
    plt.ylabel('% left actions from A')
    plt.xlabel('Episodes')
    x_ticks = np.arange(0, 301, 20)
    y_ticks = np.arange(0, 1.01, 0.05)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks, ['0%', '5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%', '55%', '60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%', '100%'])
    plt.plot(range(300), prob_E_A_left, '-', label='Double Q-Learning')
    plt.plot(np.ones(300) * 0.05, label='Optimal')
    plt.title('Comparison of the effect of 4 algorithms on Ex 6.7')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
