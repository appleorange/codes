import torch
import torch.nn as nn
import argparse,math,numpy as np
from load_data import get_data

from config_args import get_args

from pdb import set_trace as stop
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

if __name__ == "__main__":  
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using the CPU instead.")


    args = get_args(argparse.ArgumentParser())
    
    print('Labels: {}'.format(args.num_labels))

    train_loader, valid_loader, test_loader = get_data(args)

    #exit()

    num_labels = args.num_labels
    model = models.resnet18(pretrained=True)
    #The original FC layer is replaced with a new one that has num_labels outputs and a sigmoid activation function.
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, num_labels), nn.Sigmoid())  # num_labels to be defined based on your dataset
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Example scheduler, adjust as needed

    # (TODO: shall we use nn.CELoss for activiticy classification?)
    criterion = nn.BCELoss()
    num_epochs = args.epochs

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
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_accuracy = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()
            inputs, labels = batch['image'].to(device), batch['labels'].to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracy = exact_match_accuracy(outputs, labels)
            total_accuracy += accuracy

            progress_bar.set_postfix(loss=running_loss/(progress_bar.n + 1), accuracy=100. * total_accuracy/(progress_bar.n + 1))
            # Remember this is percentage
        scheduler.step()  # Update the learning rate
        save_model(epoch, model, optimizer, running_loss / len(train_loader), f"model_epoch_{epoch+1}.pth")

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100. * total_accuracy / len(train_loader):.2f}%')


    # Testing Loop
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        for batch in progress_bar:
            inputs, labels = batch['image'].to(device), batch['labels'].to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            accuracy = exact_match_accuracy(outputs, labels)
            test_accuracy += accuracy

            progress_bar.set_postfix(test_loss=test_loss/(progress_bar.n + 1), accuracy=100. * test_accuracy/(progress_bar.n + 1))

    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100. * test_accuracy / len(test_loader):.2f}%')