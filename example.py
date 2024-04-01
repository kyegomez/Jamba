import torch 
from jamba.model import JambaBlock

# Create a random tensor of shape (1, 128, 512)
x = torch.randn(1, 128, 512)

# Create an instance of the JambaBlock class
jamba = JambaBlock(
    512,  # input channels
    128,  # hidden channels
    128,  # key channels
    8,    # number of heads
    4,    # number of layers
)

# Pass the input tensor through the JambaBlock
output = jamba(x)

# Print the shape of the output tensor
print(output.shape)
