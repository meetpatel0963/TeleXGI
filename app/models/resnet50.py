import torch
import torch.nn as nn
import torchvision.models as models

from config import Config


class ResNet50Classifier(nn.Module):
    
    def __init__(self, mlp_input_dim=Config.R_MLP_INPUT_DIM, mlp_hidden_dim=Config.MLP_HIDDEN_DIM, \
                 mlp_dropout_rate=Config.MLP_DROPOUT_RATE, num_classes=Config.NUM_CLASSES):
        
        super(ResNet50Classifier, self).__init__()
        
        # Load the pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        
        # Define an MLP for classification
        self.fc = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Forward pass through ResNet-50
        resnet_features = self.resnet50(x)
        
        # Flatten the features
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        
        # Forward pass through MLP
        output = self.fc(resnet_features)
        return output

    def get_target_layers(self): 
        # Return target layers for GradCAM
        return self.resnet50[7][-1]
