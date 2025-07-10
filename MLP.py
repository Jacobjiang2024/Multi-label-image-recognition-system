import torch.nn as nn
from transformers import AutoModel

# A flexible multi-layer perceptron (MLP) head for classification
# Can be configured with arbitrary hidden layers, dropout, activation, and optional batch norm
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1,
                 activation='relu', use_batchnorm=False,use_dropout=True):
        """
            Args:
                input_dim (int): Dimension of input features
                hidden_dims (list of int): List specifying number of units in each hidden layer
                output_dim (int): Output dimension (e.g., number of classes)
                dropout_rate (float): Dropout probability after each hidden layer
                activation (str): Activation function name ('relu', 'gelu', 'tanh')
                use_batchnorm (bool): Whether to apply BatchNorm after each hidden layer
        """
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        act_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }.get(activation.lower(), nn.ReLU())

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(act_fn)
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(dims[-1], output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# A general wrapper combining a HuggingFace transformer with a customizable MLP classification head
class BaseTransformerWithHead(nn.Module):
    def __init__(self, base_model_name, output_dim, hidden_dims=None,
                 dropout_rate=0.1, activation='relu', use_batchnorm=False,
                 freeze_base=False):
        """
            Args:
                base_model_name (str): Name of the HuggingFace pretrained model (e.g., 'bert-base-uncased')
                output_dim (int): Output dimension (e.g., number of labels)
                hidden_dims (list[int], optional): Hidden layer sizes; if None, defaults to [hidden_size // 2, hidden_size // 4]
                dropout_rate (float): Dropout probability applied to each hidden layer
                activation (str): Activation function to use ('relu', 'gelu', etc.)
                use_batchnorm (bool): Whether to apply BatchNorm after hidden layers
                freeze_base (bool): If True, freezes the transformer parameters
        """
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        hidden_size = self.base.config.hidden_size
        if hidden_dims is None:
            hidden_dims = [hidden_size // 2, hidden_size // 4]

        self.classifier = MLPHead(
            input_dim=hidden_size,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batchnorm=use_batchnorm
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        return self.classifier(pooled)