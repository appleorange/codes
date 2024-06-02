import torch
import torch.nn as nn
import argparse,math,numpy as np
from load_data import get_data

from config_args import get_args

from pdb import set_trace as stop
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from utils import plot_training_results

def save_model(epoch, model, optimizer, loss, file_path="model_checkpoint.pth"):
    """
    Saves the model state along with training metadata.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, file_path)
    print(f"Model saved to {file_path}")

# (TODO we need to define another accuracy function for single-label multi-class classification)
def single_activity_accuracy(outputs, labels, debugging_details=False):
    #print(f"single_activity_accuracy: outputs.shape = {outputs.shape} labels.shape = {labels.shape}")
    #print(f"single_activity_accuracy: outputs = {outputs}")
    #print(f"single_activity_labels: labels = {labels}")
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    mismatched_labels = None
    if (debugging_details == True):
        idxs_mask = (predicted != labels.view_as(predicted)).view(-1)
        #print(f"idxs_mask = {idxs_mask}")
        mismatched_labels = []
        if idxs_mask.numel():
            mismatched_labels = labels[idxs_mask].cpu().numpy()

        # # collect the mismatched labels and outputs
        # mismatched_labels = []
        # for i in range(len(labels)):
        #     if (predicted[i] != labels[i]):
        #         mismatched_labels.append(labels[i].item())
        
    return accuracy, mismatched_labels

def exact_match_accuracy(outputs, labels, threshold=0.5):
    # Convert outputs to binary predictions based on the threshold
    predictions = (outputs > threshold).float()

    # Identify correct predictions (both true positives and true negatives are considered correct)
    correct_predictions = (predictions == labels).float()

    # Count exact matches for each sample (all labels must match)
    exact_matches = correct_predictions.prod(dim=1)

    # Calculate the total number of non-zero labels across all samples
    total_non_zero_labels = labels.sum()

    # Sum the exact matches to get the total number of completely correct samples
    num_exact_matches = exact_matches.sum()

    # Calculate the accuracy as the ratio of exact match counts over the total non-zero labels
    accuracy = num_exact_matches / total_non_zero_labels
    return accuracy.item()

def train_one_epoch(epoch_number, model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_accuracy = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        inputs, labels = batch['image'].to(device), batch['labels'].to(device).float()
        #(TODO) test whether we need to squeeze the labels for singleactivity classification
        labels = labels.squeeze(1).long()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (args.dataset == 'youhome_activity'):
            accuracy, _ = single_activity_accuracy(outputs, labels)
        else:
            accuracy = exact_match_accuracy(outputs, labels)
        total_accuracy += accuracy

        # print outputs if the batch is the first one
        # if progress_bar.n == 0:
        #     print(f"Outputs: {outputs}")
        #     print(f"Labels: {labels}")
        #     print(f"Loss: {loss}")
        #     print(f"Accuracy: {accuracy}")

        progress_bar.set_postfix(loss=running_loss/(progress_bar.n + 1), accuracy=100. * total_accuracy/(progress_bar.n + 1))
        # Remember this is percentage
    return running_loss, total_accuracy

def eval_one_epoch(epoch_number, model, eval_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_accuracy = 0.0
    progress_bar = tqdm(eval_loader, desc="Eval", leave=False)
    for batch in progress_bar:
        inputs, labels = batch['image'].to(device), batch['labels'].to(device).float()
        #(TODO) test whether we need to squeeze the labels for singleactivity classification
        labels = labels.squeeze(1).long()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        if (args.dataset == 'youhome_activity'):
            accuracy, _ = single_activity_accuracy(outputs, labels)
        else:
            accuracy = exact_match_accuracy(outputs, labels)
        total_accuracy += accuracy

        # print outputs if the batch is the first one
        # if progress_bar.n == 0:
        #     print(f"Outputs: {outputs}")
        #     print(f"Labels: {labels}")
        #     print(f"Loss: {loss}")
        #     print(f"Accuracy: {accuracy}")

        progress_bar.set_postfix(loss=running_loss/(progress_bar.n + 1), accuracy=100. * total_accuracy/(progress_bar.n + 1))
        # Remember this is percentage
    return running_loss, total_accuracy

# count the number of mismatched labels for each label
def summarize_mismatched_labels(mismatched_labels):
    mismatched_labels_dict = {}
    for label in mismatched_labels:
        if label in mismatched_labels_dict:
            mismatched_labels_dict[label] += 1
        else:
            mismatched_labels_dict[label] = 1
    print(f"Summarized mismatched labels: {mismatched_labels_dict}")
    
def run_testing(model, test_loader, criterion, device, debugging_details=False):
    # TODO(load the best model)
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0

    mismatched_labels = np.array([])
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in progress_bar:
            inputs, labels = batch['image'].to(device), batch['labels'].to(device).float()
             #(TODO) test whether we need to squeeze the labels for singleactivity classification
            labels = labels.squeeze(1).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            if (args.dataset == 'youhome_activity'):
                mismatched_labels_in_batch = []
                accuracy, mismatched_labels_in_batch = single_activity_accuracy(outputs, labels, debugging_details)
                if (mismatched_labels_in_batch is not None and len(mismatched_labels_in_batch) > 0):
                    mismatched_labels = np.concatenate((mismatched_labels, mismatched_labels_in_batch))
            else:
                accuracy = exact_match_accuracy(outputs, labels)
            test_accuracy += accuracy

            progress_bar.set_postfix(test_loss=test_loss/(progress_bar.n + 1), accuracy=100. * test_accuracy/(progress_bar.n + 1))

    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * test_accuracy / len(test_loader):.2f}%')
    if (debugging_details == True):
        print(f"Mismatched labels: {mismatched_labels}")
        summarize_mismatched_labels(mismatched_labels)

if __name__ == "__main__":  
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using the CPU instead.")

    args = get_args(argparse.ArgumentParser())
    
    # step 1: check args and compatibility
    if (args.run_testing_only == True):
        assert args.load_saved_model == True, "The model must be loaded from the saved model for testing only."
        assert args.load_from_saved_model_name != '', "The model must be loaded from the saved model for testing only."

    print('Labels: {}'.format(args.num_labels))

    # step 2: load data and check data
    
    ###train_loader, valid_loader, test_loader = get_data(args)
    train_loader, test_loader = get_data(args)

    print(f"Test loader length: {len(test_loader)}")
    
    if (args.run_testing_only == False):
        # print the length of the train, validation and test loaders
        print(f"Train loader length: {len(train_loader)}")
       
        # print samples of the train loader to check if the data is loaded correctly
        for i, data in enumerate(train_loader):
            print(f"Sample {i+1}: image shape = {data['image'].shape}, label shape = {data['labels'].shape}")
            print(f"Labels: {data['labels']}")
            if i == 2:
                break
    
    
    # display the first six images and labels in the train loader
    # fig, axs = plt.subplots(1, 6, figsize=(9, 9))
    # for i, data in enumerate(train_loader):
    #     image = data['image'][0]
    #     label = data['labels'][0]
    #     print(f"image size {image.shape}")
    #     # the image is rgb with 576 x 576 pixels in gray. Properly display it in the 9 inch by 9 inch grid.
    #     axs[i].imshow(image.permute(1, 2, 0).numpy())
    #     axs[i].set_title(f"{label}")
    #     axs[i].axis('off')
    #     if i == 5:
    #         break
    # plt.show()

    num_labels = args.num_labels

    # step 3: load or create model
    model = models.resnet18(pretrained=True)
    if (args.dataset == 'youhome_activity'):
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_labels))
    else:
        #The original FC layer is replaced with a new one that has num_labels outputs and a sigmoid activation function.
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_labels), nn.Sigmoid())  # num_labels to be defined based on your dataset

    if (args.load_saved_model == True and args.load_from_saved_model_name != ''):
        print(f"Loading model from {args.load_from_saved_model_name}")
        model.load_state_dict(torch.load(args.load_from_saved_model_name, map_location=device))
        
    model.to(device)


    # step 4: set up optimizer, scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Example scheduler, adjust as needed

    # (TODO: shall we use nn.CELoss for activiticy classification?)
    criterion = nn.CrossEntropyLoss()
    if (args.dataset == 'youhome_activity'):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()


    # step 5: train the model and run testing
    # step 5.1: run testing only if the flag is set
    if (args.run_testing_only == True):
        run_testing(model, test_loader, criterion, device, args.dump_testing_details)
        exit()

    # step 5.2: train the model and run testing
    num_epochs = args.epochs

    loss_history = []
    accuracy_history = []
    vloss_history = []
    vaccuracy_history = []

    best_loss = np.inf
    no_improvement = 0

    start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_accuracy = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
        scheduler.step()  # Update the learning rate
        save_model(epoch, model, optimizer, running_loss / len(train_loader), f"model_epoch_{epoch+1}.pth")
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {100. * running_accuracy / len(train_loader):.2f}%')
        loss_history.append(running_loss / len(train_loader))
        accuracy_history.append(100. * running_accuracy / len(train_loader))
        
        if best_loss < running_loss or best_loss - running_loss < 0.1:
            no_improvement += 1
            if no_improvement > 3:
                print("Early stopping")
                break
        else:
            no_improvement = 0

        if best_loss > running_loss:
            best_loss = running_loss
            #save_model(epoch, model, optimizer, running_loss / len(train_loader), f"best_model.pth")
            torch.save(model.state_dict(), "best_model.pth")
            #save the model to google drive with the current timestamp and epoch number as the suffix.
            if (args.save_best_model_to_gdrive == True):
                torch.save(model.state_dict(), f"/content/drive/MyDrive/sabella/research/models/best_model_{start_timestamp}")
            
            print(f"Model saved to best_model.pth")
        # # Set the model to evaluation mode, disabling dropout and using population
        # # statistics for batch normalization.
        model.eval()
        running_vloss, running_vaccuracy = eval_one_epoch(epoch, model, test_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Eval Loss: {running_vloss / len(test_loader):.4f}, Eval Accuracy: {100. * running_vaccuracy / len(test_loader):.2f}%')
        vloss_history.append(running_vloss / len(test_loader))
        vaccuracy_history.append(100. * running_vaccuracy / len(test_loader))
        # running_vloss = 0.0
        # running_accuracy = 0.0
        # # Disable gradient computation and reduce memory consumption.
        # with torch.no_grad():
        #     for i, vdata in enumerate(valid_loader):
        #         vinputs, vlabels = vdata
        #         vinputs = vinputs.to(device)
        #         vlabels = vlabels.to(device).float() 
        #         voutputs = model(vinputs)
        #         vloss = criterion(voutputs, vlabels)
        #         vaccuracy = exact_match_accuracy(voutputs, vlabels)
        #         running_vloss += vloss
        #         running_accuracy += vaccuracy

        # avg_vloss = running_vloss / (i + 1)
        # avg_vaccuracy = running_accuracy / (i + 1)
        # print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {100. * running_accuracy / len(train_loader):.2f}%, Eval Loss: {avg_vloss:.4f}, Eval Accuracy: {100. * avg_vaccuracy:.2f}%')
        # # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch+1)
        #     torch.save(model.state_dict(), model_path)
        #     print(f"Model saved to {model_path}")

    #plotting loss and accuracy history in the same plot
    #(TODO) either plot or save the data to the cloud
    
    print(f"loss_history = {loss_history}")
    print(f"accuracy_history = {accuracy_history}")
    print(f"vloss_history = {vloss_history}")
    print(f"vaccuracy_history = {vaccuracy_history}")

    if (args.save_best_model_to_gdrive == True):
        plt_save_dir = "/content/drive/MyDrive/sabella/research/models/"
        torch.save(accuracy_history, f"/content/drive/MyDrive/sabella/research/models/accuracy_history_{start_timestamp}.pt")
        torch.save(loss_history, f"/content/drive/MyDrive/sabella/research/models/loss_history_{start_timestamp}.pt")
        torch.save(vaccuracy_history, f"/content/drive/MyDrive/sabella/research/models/vaccuracy_history_{start_timestamp}.pt")
        torch.save(vloss_history, f"/content/drive/MyDrive/sabella/research/models/vloss_history_{start_timestamp}.pt")
    else:
        plt_save_dir = "./"
    plt_save_destionation = f"{plt_save_dir}loss_history_{start_timestamp}.png"
    plot_training_results(accuracy_history, vaccuracy_history, loss_history, vloss_history, "Training Loss and Accuracy History", plt_save_destionation)
    
    # step 5.3: run testing
    #load the best model saved earlier for testing
    model.load_state_dict(torch.load("best_model.pth"))
    run_testing(model, test_loader, criterion, device, args.dump_testing_details)