#lstm embedder arguments

"""
preprocessing steps:
    1. Min max scaler
    2. finer crepe 1 ms?
    3. remove samples that have too far mean range ?
"""
# path arguments
train_song_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\train_song_crepe.pkl'
train_hum_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\train_hum_crepe.pkl'
val_song_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_song_crepe.pkl'
val_hum_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_hum_crepe.pkl'


# Trainer arguments
batch_size = 64

# Model arguments
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

# dataset arguments
# song freq and hum_freq should have different normalization parameters
MIN = 20
MAX = 600
LOW = -1
HIGH = 1
scaler = lambda x: LOW + ((x - MIN)*(HIGH-LOW))/(MAX-MIN)
sample_len = 1100