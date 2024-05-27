import torch

# (TODO we need to define another accuracy function for single-label multi-class classification)
def single_activity_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

#write a test code for the single_activity_accuracy function
def test_single_activity_accuracy():
    outputs = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]])
    labels = torch.tensor([1, 0, 2, 1])
    accuracy = single_activity_accuracy(outputs, labels)
    assert accuracy == 1.0, f"Accuracy: {accuracy}"

test_single_activity_accuracy()