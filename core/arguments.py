#lstm embedder arguments
from torch.nn.functional import embedding


input_size = 1
hidden_size = 1024
num_layers = 3
dropout = 0.5
bidirectional = True
proj_size = 512

# resnet1d arguments
base_filters = 8
kernel_size = 10
stride = 2
groups = 1
n_blocks = 28
embedding_dim = 512


# training arguments
device = 'cpu'