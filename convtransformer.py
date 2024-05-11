import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import init


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size).to(device)
        position = torch.arange(0, max_len).unsqueeze(1).float().to(device)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_size)).to(device)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         print(V.shape)

#         # Create mask tensor based on V.shape
#         if mask is None:
# #             mask = torch.triu(torch.ones(V.shape[2], V.shape[2]), diagonal=1).bool()
# #             mask = mask.unsqueeze(0).unsqueeze(1)  # Add batch and head dimensions
# #             mask = mask.expand(batch_size, -1, -1, -1).to(device)  # Expand along the batch dimension


#         print(mask.shape)

        # Attention scores using scaled_dot_product_attention
        attention = F.scaled_dot_product_attention(Q, K, V,mask)

        # Reshape and concatenate
        x = attention.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.embed_size)

        # Final linear transformation
        x = self.fc(x)

        return x


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1DBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)  # You can adjust kernel_size and stride as needed

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(TransformerBlock, self).__init__()
        self.multihead_attn = MultiHeadAttention(embed_size, heads)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.LeakyReLU(0.1),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.multihead_attn(x)
        x = x + self.dropout(attn_output)  # Skip connection
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Skip connection
        x = self.layer_norm2(x)
        return x






class ConvTransformerClassifier(nn.Module):
    def __init__(self, input_size, num_classes, conv_channels, conv_kernel_sizes, conv_strides, embed_size, num_heads, num_transformer_blocks, mlp_hidden_dim, dropout):
        super(ConvTransformerClassifier, self).__init__()

        # Convolutional layers
        self.conv_blocks = nn.ModuleList([
            Conv1DBlock(9 if i == 0 else conv_channels[i - 1],
                        conv_channels[i], conv_kernel_sizes[i], conv_strides[i])
            for i in range(len(conv_channels))
        ])


        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_size, max_len=input_size)



        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout)
            for _ in range(num_transformer_blocks)
        ])

        # MLP head for classification with BatchNorm
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_size, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),  # BatchNorm added
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
        # Additional <cls> token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))


#         # Initialize linear layers with Xavier initialization
#         for layer in self.mlp_head:
#             if isinstance(layer, nn.Linear):
#                 init.xavier_uniform_(layer.weight)

        # Apply Xavier initialization to layers
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        # Convolutional feature extraction
        for conv_block in self.conv_blocks:
            x = conv_block(x)
#         print(x.shape)

        # Reshape for transformer input
        x = x.permute(0, 2, 1)
#         print(x.shape)

        # Add <cls> token at the beginning
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Apply positional encoding
        x = x + self.positional_encoding(x)

#         print(x.shape)
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

#         # Transformer blocks with checkpointing
#         for transformer_block in self.transformer_blocks:
#             x = checkpoint(transformer_block.forward, x,use_reentrant= True)

#         print(x.shape)
        # Global average pooling
        x = x.mean(dim=1)

        # MLP head for classification
        x = self.mlp_head(x)

        return x
    
# Instantiate the model
input_size = 16000
num_classes = 10
conv_channels = [32, 64, 128]
conv_kernel_sizes = [8, 5, 3]
conv_strides = [1, 1, 1]
embed_size = 128
num_heads = 4
num_transformer_blocks = 2
mlp_hidden_dim = 128
dropout = 0.1

model_2 = ConvTransformerClassifier(input_size, num_classes, conv_channels, conv_kernel_sizes, conv_strides, embed_size, num_heads, num_transformer_blocks, mlp_hidden_dim, dropout)

# Dummy input tensors
input_tensor = torch.randn((32, 9, 16000))
model_2 = model_2

output = model_2(input_tensor)
print(output.shape)
