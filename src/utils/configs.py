import torch


class TrainingConfig():
    def __init__(self):
        self.dtype = torch.float32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
