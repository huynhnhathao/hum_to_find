"""
This branch ideas:
The stride of kernel significantly affect the runing time, stride 1 run 5mins, 
    but stride 2 run 0.x min
    1. resnet1d has kernel size 5x1, embedding_dim 512, stride 1, total parameters
        blocks 26, base filter 16,
    2. input crepe is random split to 4sec chunk, only split when the dataset object
        is called, randomly choose 4sec chunk in range 11 secs on both hum and song
    
    3. when evaluating, we run many 4secs chunks embedding over the query and the song.
        for the song, we use a runing window of 4secs with hop size = 0.5sec over the
        song, then each song has multiple embeddings corresponding to its chunks.
        For the query, we run 4secs window of hop size = 0.5 over the query. Then
        for each query's chunk embedding, we search on database of song embeddings.
        Then we set a threshold of k <= 10, embeddings that in k-neighbors
        to our query embedding is a match. Then if we have many match of one query
        on one song, it probabily that song that we are looking for.
    
    4. increase the dataset len by a hack in dataset class, in order to increase the
        training time of one epoch

    5. apply L2 normalize on embeddings

    6. alpha in triplet loss is 1.0
     
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

TRAIN_ON = 'colab'
#colab path arguments
if TRAIN_ON == 'colab':
    log_file_path = '/content/drive/MyDrive/hum_project/log.txt'

    train_data_path = '/content/hum_to_find/crepe_freq/train_data.pkl'
    val_data_path = '/content/hum_to_find/crepe_freq/val_data.pkl'
    save_model_path = '/content/drive/MyDrive/hum_project'
    pretrained_model_path = None

elif TRAIN_ON == 'home':

    log_file_path = r'C:\Users\ASUS\Desktop\repositories\hum_to_find\core\log.txt'

    train_data_path = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\train_data.pkl'
    val_data_path = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_data.pkl'
    save_model_path = r'C:\Users\ASUS\Desktop\hum\data'
    pretrained_model_path = r'C:\Users\ASUS\Desktop\hum\data\model_epoch2500.pt'
    
# Trainer 
epochs = 10000
batch_size = 128 # the actual batchsize will double this
learning_rate = 0.001
eval_each_num_epochs = 1
checkpoint_epochs = 1
alpha_triplet_loss = 2.0

# Model arguments
input_size = 1
hidden_size = 1024
num_layers = 3
dropout = 0.5
bidirectional = True
proj_size = 512

# resnet1d arguments
base_filters = 8
kernel_size = 8
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
sample_len = 1000
# len for each chunk of sample in second
chunk_len = 8
hop_len = 0.5
epoch_hack = 50
# GPU usage report
# print(torch.cuda.get_device_name(0))
# print('-'*100)
# print('Memory Usage:')
# print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
# print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
