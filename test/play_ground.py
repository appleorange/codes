import torch
import torch.nn as nn

def one_hot_ce_loss(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    _, labels = torch.max(targets, dim=1)
    print(f"Labels: {labels}")
    softmax = nn.Softmax(dim=1)
    softmax_outputs = softmax(outputs)
    print(f"Softmax outputs: {softmax_outputs}")
    return criterion(outputs, labels)


targets = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.int32)
outputs = torch.rand(size=(4, 3), dtype=torch.float32)
print(outputs)
loss = one_hot_ce_loss(outputs, targets)
print(loss)