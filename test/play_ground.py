import torch
import torch.nn as nn

# def one_hot_ce_loss(outputs, targets):
#     criterion = nn.CrossEntropyLoss()
#     _, labels = torch.max(targets, dim=1)
#     print(f"Labels: {labels}")
#     softmax = nn.Softmax(dim=1)
#     softmax_outputs = softmax(outputs)
#     print(f"Softmax outputs: {softmax_outputs}")
#     return criterion(outputs, labels)


# targets = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=torch.int32)
# outputs = torch.rand(size=(4, 3), dtype=torch.float32)
# print(outputs)
# loss = one_hot_ce_loss(outputs, targets)
# print(loss)



from sklearn.utils import class_weight
import numpy as np

# y is your array of class labels
y = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

print(class_weights)