#lstm embedder arguments

"""
preprocessing steps:
    1. Min max scaler
    2. finer crepe 1 ms?
    3. remove samples that have too far mean range ?
"""
# path arguments
# train_song_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\train_song_crepe.pkl'
# train_hum_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\train_hum_crepe.pkl'
# val_song_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_song_crepe.pkl'
# val_hum_freq = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_hum_crepe.pkl'

# train_data_path = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\train_data.pkl'
# val_data_path = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_data.pkl'

# log_file_path = r'C:\Users\ASUS\Desktop\repositories\hum_to_find\core'
# save_model_path = r'C:\Users\ASUS\Desktop\repositories\hum_to_find'

#colab path arguments
log_file_path = '/content/drive/MyDrive/hum_project/log.txt'

train_data_path = '/content/hum_to_find/crepe_freq/train_data.pkl'
val_data_path = '/content/hum_to_find/crepe_freq/val_data.pkl'
save_model_path = '/content/hum_to_find'

pretrained_model_path = '/content/drive/MyDrive/hum_project/model_epoch2500.pt'

# Trainer 
epochs = 10000
batch_size = 128 # the actual batchsize will double this
learning_rate = 0.001
eval_each_num_epochs = 50
checkpoint_epochs = 50

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
device = 'cuda'

# dataset arguments
# song freq and hum_freq should have different normalization parameters
MIN = 20
MAX = 600
LOW = -1
HIGH = 1
scaler = lambda x: LOW + ((x - MIN)*(HIGH-LOW))/(MAX-MIN)
sample_len = 1100