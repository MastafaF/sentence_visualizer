# Introduction

This project aims to build for you and help you visualize in an unsupervised fashion your sentences.

You just need a dataframe with a 'txt' column where your sentences to be visualized are
gathered.

We will do the rest for you! :hug_face: 

# Data 

Data is expected to be in a tsv format with the following columns: 
 * txt: txt column contains all the sentences to visualize.
 * labels: labels columns contains all the labels of the sentences.


# Requirements

```bash
pip install requirements.txt
```

# Visualize from CLI 

If you want to focus on labels as well: 
```bash
sh embed_visualize_with_labels.sh
```

If you do not care about labels: 
```bash
sh embed_visualize.sh
```