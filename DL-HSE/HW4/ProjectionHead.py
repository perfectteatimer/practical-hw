import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()

        """
        Here you should write simple 2-layer MLP consisting:
        2 Linear layers, GELU activation, Dropout and LayerNorm. 
        Do not forget to send a skip-connection right after projection and before LayerNorm.
        The whole structure should be in the following order:
        [Linear, GELU, Linear, Dropout, Skip, LayerNorm]
        """
        self.linear1 = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.projection_dim = nn.Sequential(
            self.linear1,
            self.gelu,
            self.linear2,
            self.dropout
        )
        self.skip_proj = nn.Linear(embedding_dim, projection_dim) 
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        """
        Perform forward pass, do not forget about skip-connections.
        """
        projection = self.projection_dim(x)
        skip = self.skip_proj(x)
        result = self.layer_norm(projection + skip)
        return result