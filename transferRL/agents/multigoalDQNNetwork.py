import torch

class MultigoalDQNNetwork(torch.nn.Module):

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=out_features),
            torch.nn.Identity(),
        )
    def forward(self,x):
        x = self.features(x)
        return x