import matplotlib.pyplot as plt

# creata a myPlot function to plot two series in the same plot. The inputs are the two series and the title of the plot. The function should save the plot as "loss_history.png" in the current directory.
def plot_training_results(accuracy_history, loss_history, title):
    plt.figure()
    plt.title(title)
    # Plot the loss curves
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot the accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    plt.savefig("loss_history.png")
    return