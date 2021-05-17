import matplotlib.pyplot as plt

def plot_scatter(x, y, labels, title, xlabel, ylabel):
    """"
    Plots a scatterplot given labels and x and y.
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.scatter(x, y, c=labels, label=labels, s=.75)