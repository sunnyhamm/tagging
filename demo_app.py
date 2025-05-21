# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:29:07 2025

@author: cyy12
"""


################### this app is to visualize the entities tagged by SpaCy and NLTK row by row

import streamlit as st
import pandas as pd
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

nlp = spacy.load("en_core_web_sm")

# Download NLTK resources (one time run)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')


#  read MDR data
df = pd.read_csv('cleanMDR.csv')[['name','definition']].rename(columns={'definition':'text'})
df = df.iloc[:50]  # for app to be able to run. otherwise NLTK model takes two hours for 36k rows
print(df.shape)


# NLTK annotation function
def nltk_annotate(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags)
    ner = []
    for subtree in chunks:
        if isinstance(subtree, Tree):
            entity = " ".join([token for token, _ in subtree.leaves()])
            ner.append((entity, subtree.label()))
    return pos_tags, ner

# SpaCy annotation function
def spacy_annotate(text):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    ner = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, ner, doc

# Streamlit UI
st.title("🔍 MDR Definition (per row) Annotation Comparison: SpaCy vs NLTK")

selected_row = st.selectbox("Select a sentence to analyze", df.index, format_func=lambda x: df["text"][x])
text = df["text"][selected_row]

st.subheader("Original Sentence")
st.write(text)

# Get results
spacy_pos, spacy_ner, spacy_doc = spacy_annotate(text)
nltk_pos, nltk_ner = nltk_annotate(text)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧠 SpaCy Results")
    st.write("**POS Tags:**", spacy_pos)
    st.write("**Named Entities:**", spacy_ner)
    st.markdown("**Entity Visualization:**")
    st.markdown(spacy_doc._.rendered if hasattr(spacy_doc._, "rendered") else "", unsafe_allow_html=True)
    
    # Fallback if no displacy extension
    st.components.v1.html(spacy.displacy.render(spacy_doc, style="ent", page=True), height=300, scrolling=True)

with col2:
    st.markdown("### 🧪 NLTK Results")
    st.write("**POS Tags:**", nltk_pos)
    st.write("**Named Entities:**", nltk_ner)
    
## to run this app, go to anaconda prompt, and do: 
# streamlit run demo_app.py

# app will have an error. but dashboard works

