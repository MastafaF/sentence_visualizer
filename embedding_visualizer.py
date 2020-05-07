# 1. Read embedding file
# 2. Convert to tensorboard
# 3. Visualize

# encoding: utf-8
import sys, os
import tensorflow as tf
import numpy as np
# from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector
import logging
# from tensorboard import default
from tensorboard import program
import pandas as pd

class TensorBoardTool:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def run(self, emb_name, port):
        # Remove http messages
        # log = logging.getLogger('sonvx').setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        logging.propagate = False
        # Start tensorboard server
        # tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.dir_path, '--port', str(port)])
        url = tb.launch()
        sys.stdout.write('TensorBoard of %s at %s \n' % (emb_name, url))


def convert_multiple_emb_models_2_tf(emb_name_arr, w2v_model_arr, output_path, port):
    """
    :param emb_name_arr:
    :param w2v_model_arr:
    :param output_path:
    :param port:
    :return:
    """
    idx = 0
    # define the model without training
    sess = tf.compat.v1.InteractiveSession()
    config = projector.ProjectorConfig()

    for w2v_model in w2v_model_arr:
        emb_name = emb_name_arr[idx]

        meta_file = "%s.tsv" % emb_name
        placeholder = np.zeros((len(w2v_model.wv.index2word), w2v_model.vector_size))

        with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
            for i, word in enumerate(w2v_model.wv.index2word):
                placeholder[i] = w2v_model[word]
                # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
                if word == '':
                    print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                    file_metadata.write(u"{0}".format('<Empty Line>').encode('utf-8') + b'\n')
                else:
                    file_metadata.write(u"{0}".format(word).encode('utf-8') + b'\n')

        word_embedding_var = tf.Variable(placeholder, trainable=False, name=emb_name)
        tf.global_variables_initializer().run()
        sess.run(word_embedding_var)

        # adding into projector
        embed = config.embeddings.add()
        embed.tensor_name = emb_name
        embed.metadata_path = meta_file
        idx += 1

    # saver = tf.train.Saver()
    saver = tf.compat.v1.train.Saver()

    writer = tf.summary.FileWriter(output_path, sess.graph)

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    all_emb_name = "_".join(emb_name for emb_name in emb_name_arr)
    saver.save(sess, os.path.join(output_path, '%s.ckpt' % all_emb_name))
    # tf.flags.FLAGS.logdir = output_path
    # print('Running `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))
    # tb.run_main()q
    tb_tool = TensorBoardTool(output_path)
    tb_tool.run(all_emb_name, port)
    return

def convert_my_embed_model_2_tf(model_name, output_path, port):
    emb_name = "test_embed"
    meta_file = "%s.tsv" % emb_name

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        N_NEWS_SENTENCES = 1490
        filename_df_date_news = "../corona_news_importance/data/df_date_news.tsv"
        df_date_news = pd.read_csv(filename_df_date_news, sep='\t')

        for k in range(N_NEWS_SENTENCES):
            file_metadata.write(u"{}".format(k%2).encode('utf-8') + b'\n')

        # for sentence in df_date_news.loc[:, 'sentence']:
        #     # word = "TOTO{}".format(str(k))
        #     file_metadata.write(u"{}".format(sentence.strip()).encode('utf-8') + b'\n')

    placeholder = np.load(model_name)
    print("SHAPE {}".format(placeholder.shape))
    # define the model without training
    g = tf.Graph()
    with g.as_default():
        # build graph...
        word_embedding_var = tf.Variable(placeholder, trainable=False, name=emb_name)

    sess = tf.compat.v1.InteractiveSession(graph = g)
    # sess = tf.compat.v1.Session()

    print("TENSORFLOW SHAPE {}".format(word_embedding_var.shape))
    tf.compat.v1.global_variables_initializer().run() # Added by Mastafa additionnally

    sess.run(word_embedding_var)
    # tf.global_variables_initializer().run()

    saver = tf.compat.v1.train.Saver()
    writer = tf.compat.v1.summary.FileWriter(output_path, sess.graph) # UPDATED BY MASTAFA

    # Comprehension via https://www.easy-tensorflow.com/tf-tutorials/tensorboard/tb-embedding-visualization
    # adding into projector
    # # Create a config object to write the configuration parameters
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = emb_name
    # Link this tensor to its metadata file (e.g. labels) -> we will create this file later
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, '%s.ckpt' % emb_name))
    # tf.flags.FLAGS.logdir = output_path
    # print('Running `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))
    # tb.run_main()q
    tb_tool = TensorBoardTool(output_path)
    tb_tool.run(emb_name, port)
    return


def convert_one_emb_model_2_tf(emb_name, model, output_path, port):
    """
    :param model: Word2Vec model
    :param output_path:
    :return:
    """
    # emb_name = "word_embedding"
    meta_file = "%s.tsv"%emb_name
    placeholder = np.zeros((len(model.wv.index2word), model.vector_size))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                file_metadata.write(u"{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write(u"{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    word_embedding_var = tf.Variable(placeholder, trainable=False, name=emb_name)
    sess.run(word_embedding_var)
    # tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # Comprehension via https://www.easy-tensorflow.com/tf-tutorials/tensorboard/tb-embedding-visualization
    # adding into projector
    # # Create a config object to write the configuration parameters
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = emb_name
    # Link this tensor to its metadata file (e.g. labels) -> we will create this file later
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, '%s.ckpt'%emb_name))
    # tf.flags.FLAGS.logdir = output_path
    # print('Running `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))
    # tb.run_main()
    tb_tool = TensorBoardTool(output_path)
    tb_tool.run(emb_name, port)
    return


def visualize_multiple_embeddings_individually(paths_of_emb_models):
    output_root_dir = "../data/embedding_tf_data/"
    starting_port = 6006
    embedding_names = []
    print("Loaded all word embeddings, going to visualize ...")

    if paths_of_emb_models and paths_of_emb_models.__contains__(";"):
        files = paths_of_emb_models.split(";")
        for emb_file in files:

            embedding_name = os.path.basename(os.path.normpath(emb_file))

            tf_data_folder = output_root_dir + embedding_name

            if not os.path.exists(tf_data_folder):
                os.makedirs(tf_data_folder)

            is_binary = False

            if emb_file.endswith(".bin"):
                is_binary = True

            emb_model = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=is_binary)

            convert_one_emb_model_2_tf(embedding_name, emb_model, tf_data_folder, starting_port)

            embedding_names.append(embedding_name)

            starting_port += 1

    while True:
        print("Type exit to quite the visualizer: ")
        user_input = input()
        if user_input == "exit":
            break
    return


def visualize_multiple_embeddings_all_in_one(paths_of_emb_models, port):
    output_root_dir = "../data/embedding_tf_data/"
    starting_port = port
    embedding_names = []
    print("Loaded all word embeddings, going to visualize ...")

    embedding_name_arr = []
    w2v_embedding_model_arr = []

    if paths_of_emb_models and paths_of_emb_models.__contains__(";"):
        files = paths_of_emb_models.split(";")
        for emb_file in files:

            embedding_name = os.path.basename(os.path.normpath(emb_file))
            embedding_name_arr.append(embedding_name)

            is_binary = False

            if emb_file.endswith(".bin"):
                is_binary = True

            emb_model = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=is_binary)
            w2v_embedding_model_arr.append(emb_model)
            embedding_names.append(embedding_name)

        # print("View side-by-side word similarity of multiple embeddings at: http://Sons-MBP.lan:8089")

    all_emb_name = "_".join(emb_name for emb_name in embedding_name_arr)
    tf_data_folder = output_root_dir + all_emb_name
    if not os.path.exists(tf_data_folder):
        os.makedirs(tf_data_folder)

    convert_multiple_emb_models_2_tf(embedding_name_arr, w2v_embedding_model_arr, tf_data_folder, starting_port)

    while True:
        print("Type exit to quite the visualizer: ")
        user_input = input()
        if user_input == "exit":
            break
    return


# def visualize_multiple_embeddings(paths_of_emb_models, port):
#     """
#     API to other part to call, don't modify this function.
#     :param paths_of_emb_models:
#     :param port:
#     :return:
#     """
#     visualize_multiple_embeddings_all_in_one(paths_of_emb_models, port)


def run_embed(model, output_path, port):
    convert_my_embed_model_2_tf(model, output_path, port)
    while True:
        print("Type exit to quite the visualizer: ")
        user_input = input()
        if user_input == "exit":
            break

"""
Working fine with conda env: env_sentence_BERT 

Need to check tensorflow version because does not work with conda env: env_corona_analysis
"""
if __name__ == "__main__":
    """
    Just run `python w2v_visualizer.py word2vec.model visualize_result`
    """
    # try:
    #     model_path = sys.argv[1]
    #     output_path = sys.argv[2]
    # except Exception as e:
    #     print("Please provide model path and output path %s " % e)

    # model = Word2Vec.load(model_path)
    # model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    # convert_one_emb_model_2_tf(model, output_path)
    model_name = "../corona_news_importance/data/arr_embeddings.npy"
    output_path = "./tf_embeddings"
    port=8080
    run_embed(model_name, output_path, port)
    # convert_one_emb_model_2_tf(model, output_path)