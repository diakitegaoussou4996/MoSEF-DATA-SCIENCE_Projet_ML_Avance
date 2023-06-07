import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')



def apply_positive_indicator(df, col_name):
    positive_words = ' positive'
    df1=df.copy()
    df1[col_name] = df1[col_name].apply(lambda x: 1 if x == positive_words else 0)
    return df1

def lemmatize_word(text):
    """
    Lemmatize the tokenized words with their POS tags
    """

    lemmatizer = WordNetLemmatizer()
    pos_tags = nltk.pos_tag(text)
    lemma = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemma

def get_wordnet_pos(treebank_tag):
    """
    Map TreeBank part-of-speech tags to WordNet part-of-speech tags
    """

    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN  # default to noun if no match found



# Define the preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


def preprocess_word2vec(X_train, X_test):
    # Build the Word2Vec model
    model = Word2Vec(X_train, vector_size=50, window=5, min_count=1, workers=12)

    # Get the embedding vectors and word labels
    word_labels = model.wv.index_to_key
    embedding_vectors = [model.wv.get_vector(word) for word in word_labels]

    # Create embedding matrix
    embedding_matrix = np.array(
        list(
            map(
                lambda word: model.wv.get_vector(word)
                if model.wv.get_vector(word) is not None
                else np.zeros(50),
                word_labels,
            )
        )
    )

    # Vectorize the data using Word2Vec
    X_train = np.array(
        [
            np.mean(
                [model.wv[token] for token in doc if token in model.wv], axis=0
            )
            for doc in X_train
        ]
    )
    X_test = np.array(
        [
            np.mean(
                [model.wv[token] for token in doc if token in model.wv], axis=0
            )
            for doc in X_test
        ]
    )

    # Define the maximum sequence length
    max_length = max([len(seq) for seq in X_train + X_test])

    # Pad sequences with zeros so that they all have the same length
    X_train = pad_sequences(X_train, maxlen=max_length, padding="post", dtype="float32")
    X_test = pad_sequences(X_test, maxlen=max_length, padding="post", dtype="float32")

    # Scale the data to ensure all values are non-negative
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, word_labels, embedding_matrix, max_length

#--------------------------------------------------------------------------------------------------------------------------
def train_and_evaluate_multinomialnb_model(X_train, X_test, y_train, y_test):
    # Train the model
    print("*************** Entrainement du modèle MultinomialNB ******************")
    model = MultinomialNB()
    history1 = model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Visualize the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return model, accuracy


def evaluate_model_SVC(train_df, test_df):
    # Charger les données
    X_train = train_df['inputs_pretokenized']
    X_test = test_df['inputs_pretokenized']

    y_train = train_df['targets_pretokenized']
    y_test = test_df['targets_pretokenized']

    # Convertir les critiques de film en représentation vectorielle TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Entraîner le modèle SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Prédire les étiquettes de sentiment sur l'ensemble de test
    y_pred = svm.predict(X_test)

    # Évaluer la précision du modèle
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # Afficher la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Calculer l'aire sous la courbe ROC
    roc_auc = roc_auc_score(y_test, y_pred)
    print('ROC AUC:', roc_auc)

    # Calculer la probabilité de chaque classe
    y_prob = svm.decision_function(X_test)

    # Calculer la courbe ROC et l'aire sous la courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Afficher la courbe ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()

    
    
    
def evaluate_model_LSTMSimple(train_df, test_df)
    # Build the Word2Vec model
    model = Word2Vec(train_df['preprocessed'], vector_size=50, window=5, min_count=1, workers=12)

    # Get the embedding vectors and word labels
    word_labels = model.wv.index_to_key
    embedding_vectors = [model.wv.get_vector(word) for word in word_labels]

    # Create embedding matrix
    embedding_matrix = np.array(list(map(lambda word: model.wv.get_vector(word) if model.wv.get_vector(word) is not None else np.zeros(50), word_labels)))


    # Convert text data to sequences of indices
    train_sequences = [[word_labels.index(word) for word in doc] for doc in train_df['preprocessed']]
    test_sequences = [[word_labels.index(word) for word in doc] for doc in test_df['preprocessed']]

    # Pad sequences to make them the same length
    max_len = max([len(seq) for seq in train_sequences + test_sequences])
    train_sequences = pad_sequences(train_sequences, maxlen=max_len)
    test_sequences = pad_sequences(test_sequences, maxlen=max_len)


    # Build the LSTM model
    model = Sequential([
        Embedding(len(word_labels), 50, weights=[embedding_matrix], input_length=max_len),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model and get the history object (run more then 25 min)
    history = model.fit(train_sequences, train_df['targets_pretokenized'], 
                        validation_data=(test_sequences, test_df['targets_pretokenized']), 
                        epochs=10)

    model.summary()

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_sequences, test_df['targets_pretokenized'])
    print('Test accuracy:', test_acc)

    import matplotlib.pyplot as plt

    # Plot the training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    return model



def evaluate_model_LSTM_Multicouches(train_df, test_df):
    # Load pre-trained word2vec model
    word2vec_model = Word2Vec(train_df2_select['preprocessed'], vector_size=50, window=5, min_count=1, workers=12)

    # Define the maximum length of a text sequence
    max_length = 50

    # Tokenize the text data
    tokenizer = Tokenizer()
    to_str = lambda x: str(x)
    tokenizer.fit_on_texts(list(map(to_str, X_train)))
    X_train = tokenizer.texts_to_sequences(list(map(to_str, X_train)))
    X_test = tokenizer.texts_to_sequences(list(map(to_str, X_test)))
    X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test = pad_sequences(X_test, maxlen=max_length, padding='post')


    # Create embedding matrix
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]



    # Define the model architecture
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=max_length, weights=[embedding_matrix], trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy: %f' % (accuracy*100))

    # Plot the accuracy and loss graphs
    import matplotlib.pyplot as plt
    plt.plot(model3.history['accuracy'], label='Accuracy')
    plt.plot(model3.history['loss'], label='Loss')
    plt.legend()
    plt.show()
    model.summary()
    
    return model 