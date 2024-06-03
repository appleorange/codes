import torch
import numpy as np
import torchmetrics
from main import single_activity_accuracy


def test_single_activity_accuracy_all_correct():
    outputs = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]])
    labels = torch.tensor([1, 0, 2, 1])
    accuracy, mismatched_indices, mismatched_names = single_activity_accuracy(outputs, labels)
    print(f"Mismatched indices: {mismatched_indices}")
    assert mismatched_indices is None, f"Mismatched labels: {mismatched_indices}"
    assert mismatched_names is None, f"Mismatched names: {mismatched_names}"
    assert accuracy == 1.0, f"Accuracy: {accuracy}"


def test_single_activity_accuracy_half_correct():
    outputs = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]])
    labels = torch.tensor([1, 0, 0, 2])
    names = ["pic1", "pic2", "pic3", "pic4"]
    accuracy, mismatched_indices, mismatched_names = single_activity_accuracy(outputs, labels, debugging_details=True, image_names=names)
    print(f"Mismatched indices: {mismatched_indices}")
    assert np.array_equal(mismatched_indices, np.array([0, 2])), f"Mismatched labels: {mismatched_indices}"
    assert mismatched_names == {"pic3" : 0, 'pic4' : 2}, f"Mismatched names: {mismatched_names}"
    assert accuracy == 0.5, f"Accuracy: {accuracy}"

def test_single_activity_accuracy_half_correct_with_confusion_matrix():
    outputs = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]])
    labels = torch.tensor([1, 0, 0, 2])
    
    confusion_matrix_metric = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=3)
    accuracy, mismatched_indices, mismatched_names = single_activity_accuracy(outputs, labels, confusion_matrix_metric)
    print(f"confusion_matrix: {confusion_matrix_metric.compute()}")
    assert confusion_matrix_metric.compute().tolist() == [[1, 0, 1], [0, 1, 0], [0, 1, 0]], f"Confusion matrix: {confusion_matrix_metric.compute()}"

test_single_activity_accuracy_all_correct()

test_single_activity_accuracy_half_correct()

test_single_activity_accuracy_half_correct_with_confusion_matrix()

