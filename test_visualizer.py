# Some initial code which is the same for all the variants
import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

def register_embedding(embedding_tensor_name, meta_data_fname, log_dir):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_tensor_name
    embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir, config)

def get_random_data(shape=(100,100)):
    x = np.random.rand(*shape)
    y = np.random.randint(low=0, high=2, size=shape[0])
    return x, y

def save_labels_tsv(labels, filepath, log_dir):
    with open(os.path.join(log_dir, filepath), 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label))

LOG_DIR = 'tmp'  # Tensorboard log dir
META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
EMBEDDINGS_TENSOR_NAME = 'embeddings'
EMBEDDINGS_FPATH = os.path.join(LOG_DIR, EMBEDDINGS_TENSOR_NAME + '.ckpt')
STEP = 0

x, y = get_random_data((100,100))
register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, LOG_DIR)
save_labels_tsv(y, META_DATA_FNAME, LOG_DIR)


# Size of files created on disk: 163kB
tf.compat.v1.disable_eager_execution()
tensor_embeddings = tf.Variable(x, name=EMBEDDINGS_TENSOR_NAME)
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.Saver()
saver.save(sess, EMBEDDINGS_FPATH, STEP)
sess.close()