import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torch

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------- NEURAL NETWORK ---------------------------- #
print("Setting up the neural network...")
class MultiInputNN(nn.Module):
    def __init__(self, num_meta_features, num_classes):
        super(MultiInputNN, self).__init__()

        # Image branch (CNN)
        self.image_branch = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        for param in self.image_branch.parameters():
            param.requires_grad = False
        
        num_ftrs = self.image_branch.classifier[1].in_features
        self.image_branch.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, 128)
        )

        # Metadata branch (Fully Connected)
        self.meta_branch = nn.Sequential(
            nn.Linear(num_meta_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

        # Combined layers
        self.classifier_head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, meta):
        image_features = self.image_branch(image)
        meta_features = self.meta_branch(meta)
        combined = torch.cat((image_features, meta_features), dim=1)
        output = self.classifier_head(combined)
        return output
    
print("Neural network setup complete.")