import matplotlib.pyplot as plt


def drawPlot(Q, obstacle):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 5)  # x axis
    ax.set_ylim(5, 0)  # y axis, inverse
    ax.hlines(range(1, 5), xmin=0, xmax=5, colors='black')  # horizontal inside grid lines
    ax.vlines(range(1, 5), ymin=0, ymax=5, colors='black')  # vertical inside grid lines
    action = ['↑: ', '→: ', '↓: ', '←: ']
    for i in range(len(Q)):
        for j in range(len(Q[i])):
            color = 'red' if Q[i][j] == max(Q[i]) else 'black'
            # (row, column, ...)
            ax.text((i%5) + 0.5, int(i/5) + 0.2*(j+1), action[j]+str(round(Q[i][j], 2)), ha='center', va='center', color=color)
    for i in obstacle:
        # (the range of row, left column, right column, ...)
        ax.fill_between(range(int(i % 5), int(i % 5)+2), int(i/5), int(i/5)+1, facecolor='gray')
    ax.fill_between(range(0, 2), 0, 1, facecolor='green')
    ax.fill_between(range(4, 6), 4, 5, facecolor='yellow')
    ax.spines['left'].set_linewidth(1.5)  # border line width
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)

    ax.set_xticks([])  # hide axis scale
    ax.set_yticks([])

    plt.show()
