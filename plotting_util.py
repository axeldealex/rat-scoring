import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(x, y, labels, title, xlabel, ylabel, legend_type=0, alpha=0.5):
    """"
    Plots a scatterplot given labels and x and y.
    Returns fig and ax objects.
    """

    if legend_type == 1:
        colours = ["tab:red", "tab:blue", "tab:green"]
        states = ["NREM", "REM", "Wake"]

    fig, ax = plt.subplots()
    for label in set(labels):
        mask = np.where(labels == label)
        x_plot = x[mask]
        y_plot = y[mask]

        ax.scatter(x_plot, y_plot, label=states[label-1], c=colours[label-1], alpha=alpha, edgecolors='none')

    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return fig, ax

def plot_scatter_no_legend(x, y, labels, title, xlabel, ylabel, legend_type=0, alpha=0.5, font_size=22):
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots()
    colours = ["tab:red", "tab:blue", "tab:green"]

    for label in set(labels):
        mask = np.where(labels == label)
        x_plot = x[mask]
        y_plot = y[mask]

        ax.scatter(x_plot, y_plot, c=colours[label - 1], alpha=alpha, edgecolors='none')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.rcParams.update({'font.size': 12})
    return fig, ax