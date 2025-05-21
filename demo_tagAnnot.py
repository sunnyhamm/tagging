# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:29:07 2025

@author: cyy12
"""

import pandas as pd
import spacy
import time
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from collections import Counter
import matplotlib.pyplot as plt
import ast


#  read MDR data
df = pd.read_csv('cleanMDR.csv')[['name','definition']].rename(columns={'definition':'text'})
# df = df.iloc[:5]  # just for testing
print(df.shape)

##################################### SpaCy models ##################################
# Apply SpaCy for POS part of speech and NER name entity recognition

# download it first: python -m spacy download en_core_web_sm

# Model	Components Included	Size	Notes
# en_core_web_sm	tagger, parser, ner, lemmatizer	~12MB	No vectors
# en_core_web_md	+ word vectors	~43MB	Good tradeoff
# en_core_web_lg	+ larger vectors	~741MB	Higher accuracy
# en_core_web_trf	Uses transformer (like BERT) for better accuracy  --- not running this for now


model = "en_core_web_sm" # pick this for its lightweight and fast performance
print(f'\n running {model} ...')

start = time.time()

nlp = spacy.load(model)

# Function to extract POS and NER using SpaCy
def spacy_analysis(text):
    doc = nlp(text)
    pos = [(token.text, token.pos_) for token in doc]
    ner = [(ent.text, ent.label_) for ent in doc.ents]
    return pd.Series([pos, ner])

# Apply to DataFrame, this takes about 2 minutes
df[['spacy_pos', 'spacy_ner']] = df['text'].apply(spacy_analysis)

# export the results, no need to rerun
f_export = 'results_SpaCy.csv'
df.to_csv(f_export, index=False)
print(f'\n{f_export}_created')


end = time.time()
duration = end - start
minutes = int(duration // 60)
seconds = duration % 60
print(f"\n SpaCy Program finished: {minutes} min {seconds:.2f} sec")


###################################### NLTK for POS and NER ##################################


start = time.time()

# # Download NLTK resources (one time run)
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

# Function to extract POS and NER using NLTK
def nltk_analysis(text):
    tokens = word_tokenize(text)
    pos = pos_tag(tokens)
    chunked = ne_chunk(pos)
    ner = []
    for subtree in chunked:
        if isinstance(subtree, Tree):
            entity = " ".join([token for token, _ in subtree.leaves()])
            ner.append((entity, subtree.label()))
    return pd.Series([pos, ner])

# Apply to DataFrame -----------------  THIS TAKES TWO HOURS !! -----------------
df[['nltk_pos', 'nltk_ner']] = df['text'].apply(nltk_analysis)

# export, no need to rerun
f_export = 'results_NLTK.csv'
df.to_csv(f_export, index=False)
print(f'\n{f_export}_created')


end = time.time()
duration = end - start
minutes = int(duration // 60)
seconds = duration % 60
print(f"\n NLTK Program finished: {minutes} min {seconds:.2f} sec")


############# plot the entities tagged for comparision ##################
def count_labels(ner_column):
    return Counter(label for ents in ner_column for _, label in ents)

df1 = pd.read_csv('results_NLTK.csv')
df2 = pd.read_csv('results_SpaCy.csv')
df = pd.merge(df1, df2, on=['name', 'text'], how='inner')

# in order to run the function count_labels(), columns need to be tuples or lists,
## sometimes, columns can look like lists but it's actually string
## Apply literal_eval safely to convert string to list of tuples
df['spacy_ner'] = df['spacy_ner'].apply(ast.literal_eval)
df['nltk_ner'] = df['nltk_ner'].apply(ast.literal_eval)

spacy_counts = count_labels(df['spacy_ner'])
nltk_counts = count_labels(df['nltk_ner'])

# Convert to DataFrame for plotting
ner_df = pd.DataFrame([spacy_counts, nltk_counts], index=['SpaCy', 'NLTK']).fillna(0).T

# Plot
ner_df.plot(kind='bar', figsize=(8, 5))
plt.title("Named Entity Type Comparison")
plt.ylabel("Count")
plt.xlabel("Entity Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Save it to a file
plt.savefig("MDR_definition_Entities.png")  # Saves as PNG
print("\n plot exported. ALL DONE")

# python demo_tagAnnot.py > log.txt  ######### if running in anaconda prompt on windows machines, make sure computer doesn't go to sleep so program finishes




