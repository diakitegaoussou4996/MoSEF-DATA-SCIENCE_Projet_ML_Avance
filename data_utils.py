#------------------- Import des packages et modules --------------------

from collections import defaultdict
import datetime
import io
import os
import re
import string
import time

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from PIL import Image
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve
from tensorflow.keras.layers import (BatchNormalization, Dense, Dropout,
Flatten, Input, InputLayer, Layer,
LayerNormalization)
from tensorflow.keras.losses import (BinaryCrossentropy,
CategoricalCrossentropy,
SparseCategoricalCrossentropy)
from tensorflow.keras.metrics import (Accuracy, CategoricalAccuracy,
SparseCategoricalAccuracy,
TopKCategoricalAccuracy)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from transformers import (BertTokenizer, BertTokenizerFast,
DataCollatorWithPadding, TFBertForSequenceClassification,
TFBertModel, create_optimizer, RobertaTokenizerFast)

import nltk
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('punkt')
nltk.download('words')


#---------------------------------------

from datasets import load_dataset
import pandas as pd

def load_imdb_data():
    
    '''
    Importation des données
    '''
    # Charger les données dans un DatasetDict
    dataset_dict = load_dataset("bigscience/P3",'imdb_Movie_Expressed_Sentiment_2')

    # Charger les données de l'ensemble d'entraînement dans un dataframe
    train_df = pd.DataFrame(dataset_dict['train'].to_pandas())

    # Charger les données de l'ensemble de test dans un dataframe
    test_df = pd.DataFrame(dataset_dict['test'].to_pandas())
    
    return train_df, test_df



def infos_data(train_df, test_df):
    '''
    Cette fonction nous permet d'afficher toutes les informations basiques de nos bases de données.
    '''
   
    # Sélection des variables 
    print("Sélection de variables : --------> OK")
    train = train_df[['inputs_pretokenized', 'targets_pretokenized']]
    test  = test_df[['inputs_pretokenized', 'targets_pretokenized']]
    print("\n")

    # Dimensions des bases de données 
    print("*********Dimensions des bases de données********")
    print("---------------  Train ------------------------")
    print(train.shape)

    print("---------------  Test ------------------------")
    print(test.shape)
    print("\n")

    # Infos sur la base de données*
    print("********** Infos sur les bases de données ********")
    print("---------------  Train ------------------------")
    train.info()
    print("\n")

    print("---------------  Test ------------------------")
    train.info()
    print("\n")

    # Nombre de modalités de chaque variable
    print("********** Nombre de modalités de chaque variable ********")
    print("---------------  Inputs ------------------------")
    print("---------------  Train ------------------------")
    print("Train :", train['inputs_pretokenized'].nunique())
    print("---------------  Test ------------------------")
    print("Test :", test['inputs_pretokenized'].nunique())
    print("\n")
    print("---------------  Target ------------------------")
    print("---------------  Train ------------------------")
    print("Train :", train['targets_pretokenized'].value_counts())
    print("---------------  Test ------------------------")
    print("Test :", test['targets_pretokenized'].value_counts())
    print("\n")

    # Affichage de quelques lignes 
    print("********** Affichage de quelques lignes ********")
    print(train['inputs_pretokenized'][0])
    print("\n")
    print(train['inputs_pretokenized'][7])
    print("\n")
    print(train['inputs_pretokenized'][145])
    
    return train, test


def remove_phrase(df, phrase):
    '''
    Cette étape nous permet de supprimer la phrase inutile qui se trouve dans notre dataFrame.
    '''
    for i in range(df.shape[0]):
        df.loc[i:i, 'test'] = df.loc[i:i, 'inputs_pretokenized'].str.startswith(phrase)

    for i in range(df.shape[0]):
        if df.loc[i, 'test'] == True: 
            df.loc[i, 'inputs_pretokenized'] = df.loc[i, 'inputs_pretokenized'].replace(phrase, '')

    df = df.drop('test', axis=1)
    return df


def compute_stats(text):
    '''
    Cette fonction permet de calculer le nombre de phrases, le nombre moyen de mots par phrase, le nombre total de mots et le nombre de mots uniques.
    '''
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    num_words = 0
    unique_words = set()
    for sentence in sentences:
        words = word_tokenize(sentence)
        num_words += len(words)
        unique_words.update(words)
    num_unique_words = len(unique_words)
    avg_words_per_sentence = num_words / num_sentences
    return pd.Series([num_sentences, avg_words_per_sentence, num_words, num_unique_words])


def process_text(text):
    '''
    
    '''
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www.\S+", "", text)
    # Remove html tags
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    text = re.sub(html, "", text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    return text


def tokenize_and_remove_stopwords(df):
    '''
    
    '''

    # Tokenize words
    df['tokenized'] = df['inputs_pretokenized'].apply(word_tokenize)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word not in stop_words])

    return df



def frequence_des_mots(data, text_column_name):
    '''
    
    '''
    
    stop_words = nltk.corpus.stopwords.words('english')
    stemmer = SnowballStemmer('english')
    
    # Create a list of all words in the text column
    all_words = data[text_column_name].apply(lambda text: ' '.join([w for w in text.split() if len(w.split()) <= 4]))
    all_words = all_words.apply(lambda text: nltk.word_tokenize(text))
    all_words = all_words.apply(lambda tokens: [w.lower() for w in tokens if w.lower() not in stop_words and w.lower() not in string.punctuation])
    all_words = all_words.apply(lambda tokens: [stemmer.stem(w) for w in tokens])
    all_words = all_words.tolist()
    all_words = [word for sublist in all_words for word in sublist] # Flatten list
    
    # Count the frequency of each word in the list
    freq = defaultdict(int)
    for word in all_words:
        freq[word] += 1
    
    # Sort the words by frequency in descending order
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    # Take only the top 100 most frequent words
    top_words = sorted_freq[:100]
    words = [w[0] for w in top_words]
    word_counts = [w[1] for w in top_words]
    
    # Create an interactive bar chart using plotly
    fig = go.Figure([go.Bar(x=words, y=word_counts)])
    fig.update_layout(
        title="Word frequencies",
        xaxis_title="Words",
        yaxis_title="Counts",
        xaxis_tickangle=-45,
        bargap=0.1,
    )
    fig.show()
    
    stats = {'total': len(all_words), 'unique': len(set(all_words))}
    
    return 


def generate_wordclouds(df):
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

    data_pos = df[df['targets_pretokenized']==' positive']
    data_neg = df[df['targets_pretokenized']==' negative']

    additional_stopwords = {"film", "movie", "one", "character", "show", "look", "seen", "scene", "see", "characters", "even", "scenes", "say", "really", "time", "still","make", "first", "watch", "director"
                           }
    stop_words.update(additional_stopwords)

    titles = ["Positif", "Negatif"]
    tables = [data_pos, data_neg]

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs = axs.ravel()

    for i, t in enumerate(tables):
        text = " ".join(review for review in t["inputs_pretokenized"])
        text = " ".join(word for word in text.split() if word not in stop_words)

        wordcloud = WordCloud().generate(text)

        axs[i].imshow(wordcloud, interpolation='bilinear')
        axs[i].axis("off")
        axs[i].set_title(titles[i])

    plt.tight_layout()
    plt.show()


# Fonction pour extraire les noms propres
def extract_entities(text):
    stop_words = set(stopwords.words("english"))
    entities = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [w for w in words if not w in stop_words]
        tagged = pos_tag(words)
        named_entities = ne_chunk(tagged)
        for entity in named_entities:
            if hasattr(entity, 'label') and entity.label() == 'PERSON':
                entities.append(' '.join(c[0] for c in entity.leaves()))
    return entities

# Fonction pour extraire les adjectifs
def extract_adjectives(text):
    '''
    '''
    stop_words = set(stopwords.words("english"))
    adjectives = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [w for w in words if not w in stop_words]
        tagged = pos_tag(words)
        for i in range(len(tagged)):
            word = tagged[i][0]
            tag = tagged[i][1]
            if tag.startswith('JJ'):  # Vérifier si le tag commence par 'JJ', qui est l'abréviation de l'adjectif en anglais
                if i > 0 and tagged[i-1][1].startswith('RB'):  # Vérifier si l'adjectif est précédé d'un adverbe (par exemple, "very good")
                    adjectives.append(tagged[i-1][0] + ' ' + word)
                else:
                    adjectives.append(word)
    return adjectives


def create_wordcloud(entities):
    '''
    '''
    # Créer un nuage de mots pour les noms cités dans les commentaires négatifs
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(entities))

    # Afficher le nuage de mots pour les commentaires négatifs
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()



#-----------------------------------------------------------------------------------------

def apply_positive_indicator(df, col_name):
    positive_words = ' positive'
    df1=df.copy()
    df1[col_name] = df1[col_name].apply(lambda x: 1 if x == positive_words else 0)
    return df1
