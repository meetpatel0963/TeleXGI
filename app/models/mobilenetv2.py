import torch
import torch.nn as nn
import torchvision.models as models

from config import Config


class MobileNetClassifier(nn.Module):
    def __init__(self, mlp_input_dim=Config.M_MLP_INPUT_DIM, mlp_hidden_dim=Config.MLP_HIDDEN_DIM, mlp_dropout_rate=Config.MLP_DROPOUT_RATE, num_classes=Config.NUM_CLASSES):
        super(MobileNetClassifier, self).__init__()
        
        # Load the pre-trained MobileNet V2 model
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Remove the final classification layer
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        self.reduce = nn.AdaptiveAvgPool2d(1)
        
        # Define an MLP for classification
        self.fc = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=mlp_dropout_rate),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Forward pass through MobileNet V2
        mobilenet_features = self.mobilenet(x)
        
        mobilenet_features = self.reduce(mobilenet_features)
        
        # Flatten the features
        mobilenet_features = mobilenet_features.view(mobilenet_features.size(0), -1)
        
        # Forward pass through MLP
        output = self.fc(mobilenet_features)
        return output

    def get_target_layers(self): 
        # Return target layers for GradCAM
        return self.mobilenet[0][18][-1]
