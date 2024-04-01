[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Jamba
PyTorch Implementation of Jamba: "Jamba: A Hybrid Transformer-Mamba Language Model"


## install
`$ pip install jamba`

## usage

```python
import torch 
from jamba.model import JambaBlock

# Create a random tensor of shape (1, 128, 512)
x = torch.randn(1, 128, 512)

# Create an instance of the JambaBlock class
jamba = JambaBlock(
    512,  # dim
    128,  # d_state
    128,  # d_con
    8,    # number of experts
    4,    # number of experts per token
)

# Pass the input tensor through the JambaBlock
output = jamba(x)

# Print the shape of the output tensor
print(output.shape)
```

# License
MIT
