import matplotlib.pyplot as plt

# creata a myPlot function to plot two series in the same plot. The inputs are the two series and the title of the plot. The function should save the plot as "loss_history.png" in the current directory.
def myPlot(accuracy_history, loss_history, title):
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

accuracy_history = [1, 21, 30, 60, 80]
loss_history = [1.5, 1.1, 0.5, 0.2, 0.1]
title = "Training Loss and Accuracy History"
myPlot(accuracy_history, loss_history, title)