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
import argparse



from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler

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



# Get sentence embeddings with SBERT
def get_sentence_embeddings(df_filename, output_filename, is_multilingual = False):
    """
    @TODO: in the future, add argument multilingual to show multilingual embeddings if needed
    :param df_filename: filename of dataframe df
    df columns: 'txt' with text sentences to embed
    :return:
    """
    print ("Using Multilingual resources? {}".format(is_multilingual))
    print( "Getting embeddings for your sentences...")
    if is_multilingual:
        embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    else:
        embedder = SentenceTransformer('bert-large-nli-mean-tokens')
    df = pd.read_csv(df_filename, sep='\t')
    arr_line = np.array(df.loc[:, 'txt'])
    embeddings = embedder.encode(arr_line) # list: [embed_1, embed_2, ...., embed_N]
    #  pour tout i, embed_i.shape = (512,)
    # @TODO: go from list(embeddings) to array of shape (N_lines, dim_embed) = (3003, 512)
    embeddings_arr = [list(embed) for embed in embeddings] # list of lists of length dim_embed
    # Store embedding in an array
    np_embed = np.array(embeddings_arr) # shape = (N_lines, dim_embed
    # save numpy array in memory
    np.save(file = output_filename, arr = np_embed)
    print("Done! ")


def convert_my_embed_model_2_tf(arr_embed_filename, df_filename, output_path, port):
    """

    :param arr_embed_filename: corresponds to the filename of the numpy array with the embeddings
    built with get_sentence_embeddings
    :param df_filename: filename of dataframe with sentences
    :param output_path:
    :param port:
    :return:
    """
    emb_name = "test_embed"
    meta_file = "%s.tsv" % emb_name

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:

        df = pd.read_csv(df_filename, sep='\t')
        # nb_sentences = df_date_news.shape[0]

        for sentence in df.loc[:, 'txt']:
            # word = "TOTO{}".format(str(k))
            file_metadata.write(u"{}".format(sentence.strip()).encode('utf-8') + b'\n')

    placeholder = np.load(arr_embed_filename)
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




def run_embed(arr_embed_filename, df_filename, output_path, port):
    convert_my_embed_model_2_tf(arr_embed_filename, df_filename, output_path, port)
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
    parser = argparse.ArgumentParser(description='Embed your data and visualize it!')
    parser.add_argument('--df_filename', type=str, default=None,
                        help="File name of your dataframe. Sentences must be in the 'txt' column")


    # We want to set a boolean to True or False which is pretty tricky actually
    # Check https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument('--multilingual', type=str, default="False",
                        help='Is your data containing multiple languages?')

    args = parser.parse_args()
    df_filename = args.df_filename


    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    IS_MULTILINGUAL = args.multilingual
    IS_MULTILINGUAL = str2bool(IS_MULTILINGUAL)

    # IS_MULTILINGUAL = False
    # df_filename = "./data/df_test.tsv"
    arr_embed_filename = "./data/arr_embed.npy"
    output_path = "./tf_embeddings"
    port=8080
    # We get sentence embeddings and save array of embedding in file arr_embed_filename
    get_sentence_embeddings(df_filename, arr_embed_filename, is_multilingual=IS_MULTILINGUAL)
    # We visualize everything on TensorBoard
    run_embed(arr_embed_filename = arr_embed_filename , df_filename = df_filename, output_path = output_path , port = port)
