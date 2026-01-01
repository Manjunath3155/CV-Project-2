import torch
import torch.nn as nn
import torchvision.models as models

class RecipeResNet(nn.Module):
    """
    Custom ResNet-50 architecture fine-tuned for the Recipe 1M+ dataset.
    Backbone: Pre-trained ResNet-50
    Head: Custom Fully Connected layers for ingredient extraction.
    """
    def __init__(self, num_classes=1000, dropout_rate=0.5):
        super(RecipeResNet, self).__init__()
        
        # Load the pre-trained ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze early layers (Layers 1-3) to preserve generic feature extraction
        # This is the 'Transfer Learning' part
        for name, child in self.backbone.named_children():
            if name in ['layer1', 'layer2', 'layer3']:
                for param in child.parameters():
                    param.requires_grad = False
        
        # Replace the final classification head
        # Recipe 1M+ subset has 10,000 images but we map to a large ingredient vocabulary
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes) # num_classes = unique ingredient tokens
        )

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """
        Extracts global average pooling features before the FC layer.
        Used for similarity-based recipe retrieval.
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

