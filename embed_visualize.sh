#!/bin/bash
# Copyright Mastafa Foufa
# mastafa.foufa@hotmail.com

#------------------------------
# Usage: sh embed_visualize.sh --df_filename DF_FILENAME --multilingual IS_MULTILINGUAL

df_filename=$1
is_multilingual=$2



echo "Embedding your sentences and visualizing them on Tensorboard..."
python embed_and_visualize.py --df_filename ${df_filename} --multilingual ${is_multilingual}