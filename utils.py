import matplotlib.pyplot as plt

# creata a myPlot function to plot two series in the same plot. The inputs are the two series and the title of the plot. The function should save the plot as "loss_history.png" in the current directory.
def plot_training_results(accuracy_history, vaccuracy_history, loss_history, vloss_history, title, plt_save_destination):
    plt.figure()
    plt.title(title)
    # Plot the loss curves
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label="Train Loss")
    plt.plot(vloss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.ylim(0, 4)

    # Plot the accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label="Train Accuracy")
    plt.plot(vaccuracy_history, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.savefig(plt_save_destination)
    plt.show()
    return


#write a test code for the myPlot function
def test_plot_training_results():
    accuracy_history = [1, 21, 30, 60, 80]
    vaccuracy_history = [1, 25, 35, 65, 80]
    loss_history = [1.5, 1.1, 0.5, 0.2, 0.1]
    vloss_history = [2.0, 1.5, 0.5, 0.4, 0.2]
    title = "Training Loss and Accuracy History"
    plt_save_destination = "test_loss_history.png"
    plot_training_results(accuracy_history, vaccuracy_history, loss_history, vloss_history, title, plt_save_destination)


#test_plot_training_results()