import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        """
        Create the model and set its weights frozen. 
        Use Transformers library docs to find out how to do this.
        """
        if pretrained: # если модель предобученная
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name) # если нетто создаем конфигурацию 
            self.model = AutoModel(config=config)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False # замораживаем параметры модели если не тренируем
        # use the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        """
        Pass the arguments through the model and make sure to return CLS token embedding
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]