import matplotlib.pyplot as plt


def DrawAverageScore(n_episodes, average_score):
    episodes = range(n_episodes)

    plt.plot(episodes, average_score, label='Average Score')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.title('Average Score Trend')
    plt.legend()
    plt.grid(True)
    plt.show()
