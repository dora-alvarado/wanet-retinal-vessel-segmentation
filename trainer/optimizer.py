import torch
from .adamwr.adamw import AdamW

def optimizer_func(optim, model, learning_rate, weight_decay = 1e-5, momentum= 0.9):
    return dict(
        SGD_Momentum = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum),
        SGD = torch.optim.SGD(model.parameters(), lr=learning_rate),
        Adam = torch.optim.Adam(model.parameters(), lr=learning_rate),
        AdamW = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        )[optim]