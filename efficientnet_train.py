import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from Trainer import Trainer

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TransferLearningNetwork(nn.Module):
    def __init__(self):
        super(TransferLearningNetwork, self).__init__()

        # Load pre-trained ResNet model
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT).to(device)

        # Freeze the parameters of the pre-trained ResNet
        for param in model.parameters():
            param.requires_grad = False

        # Modify the last layer to match the number of classes
        num_features = model.classifier[1].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, 1), nn.Sigmoid()]) # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features) # Replace the model classifier

        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x.view(-1)
    
model = TransferLearningNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainer = Trainer(data_path="/dtu/datasets1/02514/hotdog_nothotdog/",
                 image_size=512,
                 batch_size=64,
                 normalize=True)

out_dict = trainer.train(model, optimizer, num_epochs=30)
torch.save(model.state_dict(), "/zhome/bd/4/181258/02514/models/efficientnet_b7_512_64")