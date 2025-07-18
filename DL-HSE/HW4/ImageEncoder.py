import torch.nn as nn
import timm 

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector.
    """
    def __init__(
        self, model_name="resnet50", pretrained=True, trainable=False
        ):
        """
        We will use standard pretrained ResNet50, and set freeze its parameters.
        Look the documentation of TIMM on how to donwload the model: https://timm.fast.ai/
        """
        super().__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=0)

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)



