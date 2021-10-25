import torch
import torch.nn.functional as F


def accuracy(y_pred, y_true):
    with torch.no_grad():
        y_ = F.softmax(y_pred.data, 1)
        out = torch.argmax(y_, 1)

        correct = torch.eq(out.type(y_true.type()), y_true).view(-1)
        num_correct = torch.sum(correct).item()
        num_examples = correct.shape[0]
    return num_correct/num_examples
