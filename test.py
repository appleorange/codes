import torch
import numpy as np
from main import single_activity_accuracy


def test_single_activity_accuracy_all_correct():
    outputs = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]])
    labels = torch.tensor([1, 0, 2, 1])
    accuracy, mismatched_labels = single_activity_accuracy(outputs, labels)
    print(f"Mismatched labels: {mismatched_labels}")
    assert accuracy == 1.0, f"Accuracy: {accuracy}"


def test_single_activity_accuracy_half_correct():
    outputs = torch.tensor([[0.1, 0.6, 0.3], [0.6, 0.2, 0.2], [0.2, 0.2, 0.6], [0.3, 0.4, 0.3]])
    labels = torch.tensor([1, 0, 0, 2])
    accuracy, mismatched_labels = single_activity_accuracy(outputs, labels, debugging_details=True)
    print(f"Mismatched labels: {mismatched_labels}")
    assert np.array_equal(mismatched_labels, np.array([0, 2])), f"Mismatched labels: {mismatched_labels}"
    assert accuracy == 0.5, f"Accuracy: {accuracy}"

test_single_activity_accuracy_all_correct()

test_single_activity_accuracy_half_correct()

