import pandas as pd


sentences = ['toto {}'.format(str(i)) for i in range(1000)]
df = pd.DataFrame()
df['txt'] = sentences
df.to_csv("./data/df_test.tsv", sep='\t')