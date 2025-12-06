# %% [markdown] cell 0
# ### ***Link To drive***
# %% code cell 1
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] cell 2
# ### ***Import Libraries***
# %% code cell 3
!pip install spacy
!python -m spacy download en_core_web_sm
import spacy

# %% code cell 4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk


# %% [markdown] cell 5
# ### ***Load Dataset***
# %% code cell 6

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/mbti_1.csv")

# %% [markdown] cell 7
# ### ***Preprocessing***
# %% code cell 8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk # Make sure to import nltk

from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
from nltk.tokenize import word_tokenize
# Download the 'punkt' resource
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Function to preprocess text with lemmatization and stemming
def preprocess_text_full(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters, digits, and punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Stemming
    stemmed = [stemmer.stem(word) for word in lemmatized]
    # Join tokens back into string
    return ' '.join(stemmed)

# Apply full preprocessing to the 'text' column
df['full_preprocessed_text'] = df['posts'].apply(preprocess_text_full)


# %% code cell 9
posts_split = df['posts'].str.split('\|\|\|')

# %% code cell 10
df.head()

# %% code cell 11
# Preprocess the text data for Conscientiousness (J/P) trait
# Extract the Judging/Perceiving letter from the MBTI type column
# MBTI format: I/E - N/S - F/T - J/P (4th character = J or P)
df['JP'] = df['type'].apply(lambda x: x[3])
# Map J -> 0, P -> 1 for binary classification
df['class'] = df['JP'].apply(lambda x: 0 if x == 'J' else 1)

# %% [markdown] cell 12
# ### ***Machine Learning***
# %% [markdown] cell 13
# ### ***Apply POS TAGGING***
# %% code cell 14
# Load the model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to perform POS tagging
def pos_tagging(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc]
    return " ".join(pos_tags)


# %% code cell 15
# Apply POS tagging to the posts
df['pos_tags'] = df['posts'].apply(pos_tagging)

# Features and labels for Conscientiousness (J vs P)
X_ns = df['pos_tags']
y_ns = df['JP']

# %% code cell 16
# Feature extraction for all four traits
# Step 2: Feature Extraction
posts_combined = posts_split.apply(lambda x: ' '.join(x))
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_ns = tfidf_vectorizer.fit_transform(posts_combined)

# %% code cell 17
from sklearn.model_selection import train_test_split

# %% code cell 18
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import tqdm

# %% code cell 19
# Train-test split
X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X_ns, y_ns, test_size=0.2, random_state=42)

# %% code cell 20
from sklearn.feature_extraction.text import TfidfVectorizer
!pip install xgboost
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_ns = le.fit_transform(y_train_ns)
y_test_ns = le.transform(y_test_ns)

# %% code cell 21
# Define models
models = {
    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression()
}

# %% code cell 22
# Function to train and evaluate models for a given trait
def train_and_evaluate_model(X_train, X_test, y_train, y_test, trait_name):
    print(f"Training models for {trait_name}...")
    for name, model in models.items():
        print(f"Training {name} for {trait_name}...")
        # Initialize the TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        # Convert POS-tagged text to numerical features for training and testing data
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name} for {trait_name}: {accuracy}")
        print(f"Classification report for {name} for {trait_name}:")
        print(classification_report(y_test, y_pred))
        print("----------------------------------------------------")




# %% code cell 23
train_and_evaluate_model(X_train_ns, X_test_ns, y_train_ns, y_test_ns, "Intuition vs. Sensing")

# %% code cell 24
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Initialize a figure for the combined ROC curve
plt.figure(figsize=(10, 8))

# Initialize arrays to store combined false positive rates and true positive rates
all_fpr = np.linspace(0, 1, 100)
mean_tpr = 0.0


# Plot ROC curve for each classifier and calculate the mean true positive rate
for name, model in models.items():
    # Fit the model
    model.fit(X_train_vectorized, y_train_ns)

    # Get scores (decision function output) on the test set
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test_vectorized)
    else:
        scores = model.predict(X_test_vectorized)

    # Convert scores into probabilities
    y_pred_proba = (scores - scores.min()) / (scores.max() - scores.min())

    # Compute ROC curve and ROC area for Introversion vs. Extroversion (IE) trait
    fpr, tpr, _ = roc_curve(y_test_ns, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve for the model
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

    # Calculate mean true positive rate
    mean_tpr += np.interp(all_fpr, fpr, tpr)


# Calculate the mean true positive rate across all classifiers
mean_tpr /= len(models)
mean_auc = auc(all_fpr, mean_tpr)

    # Plot the combined ROC curve
plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', lw=2, label=f'Combined ROC (AUC = {mean_auc:.2f})')

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined Receiver Operating Characteristic (ROC) Curve for N/S POS tagging')
plt.legend(loc="lower right")

# Show plot
plt.grid(True)
plt.show()



# %% code cell 25
from sklearn.naive_bayes import MultinomialNB

# Initialize Naive Bayes classifier
model = MultinomialNB()

# %% code cell 26
accuracy_ns = accuracy_score(y_test_ns, y_pred_ns)

print("\nIntuition vs. Sensing:")
print(f"Accuracy: {accuracy_ns}")
print(classification_report(y_test_ns, y_pred_ns))

# %% code cell 27
# Function to train and evaluate models for a given trait
def train_and_evaluate_model(X_train, X_test, y_train, y_test, trait_name):
    print(f"Training models for {trait_name}...")
    for name, model in models.items():
        print(f"Training {name} for {trait_name}...")
        # Initialize the TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        # Convert POS-tagged text to numerical features for training and testing data
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)

        model.fit(X_train_vectorized, y_train)
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name} for {trait_name}: {accuracy}")
        print(f"Classification report for {name} for {trait_name}:")
        print(classification_report(y_test, y_pred))
        print("----------------------------------------------------")

# %% code cell 28


# %% [markdown] cell 29
# ### ***TF-IDF***
# %% code cell 30
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import pandas as pd
import tqdm

# %% code cell 31
# Feature extraction for all four traits
# Step 2: Feature Extraction
posts_combined = posts_split.apply(lambda x: ' '.join(x))
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_ns = tfidf_vectorizer.fit_transform(posts_combined)

# %% code cell 32
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_ns = le.fit_transform(y_train_ns)
y_test_ns = le.transform(y_test_ns)

# %% code cell 33
# Define models
models = {

    'SVM': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression()
}

# %% code cell 34
import numpy as np

# List to store accuracy scores for all models across four traits
accuracy_scores = []

# Function to train and evaluate models for a given trait
def train_and_evaluate_model(X_train, X_test, y_train, y_test, trait_name):
    print(f"Training models for {trait_name}...")
    for name, model in models.items():
        print(f"Training {name} for {trait_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name} for {trait_name}: {accuracy}")
        print(f"Classification report for {name} for {trait_name}:")
        print(classification_report(y_test, y_pred))
        print("----------------------------------------------------")

# %% code cell 35
train_and_evaluate_model(X_train_ns, X_test_ns, y_train_ns, y_test_ns, "Intuition vs. Sensing")


# %% code cell 36
!pip install scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# %% code cell 37
for model in models.values():
    model.probability = True

# %% code cell 38
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Initialize a figure for the combined ROC curve
plt.figure(figsize=(10, 8))

# Initialize arrays to store combined false positive rates and true positive rates
all_fpr = np.linspace(0, 1, 100)
mean_tpr = 0.0

# Plot ROC curve for each classifier and calculate the mean true positive rate
for name, model in models.items():
    model.fit(X_train_ns, y_train_ns)
    y_score = model.predict_proba(X_test_ns)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_ns, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    mean_tpr += np.interp(all_fpr, fpr, tpr)

# Calculate the mean true positive rate across all classifiers
mean_tpr /= len(models)
mean_auc = auc(all_fpr, mean_tpr)

# Plot the combined ROC curve
plt.plot(all_fpr, mean_tpr, color='black', linestyle='--', lw=2, label=f'Combined ROC (AUC = {mean_auc:.2f})')

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined Receiver Operating Characteristic (ROC) Curve- N/S TF-IDF ')
plt.legend(loc="lower right")

# Show plot
plt.grid(True)
plt.show()


# %% [markdown] cell 39
# ### ***EDA***
# %% code cell 40
df['full_preprocessed_text'] = df['posts'].apply(preprocess_text_full)

# %% code cell 41
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Filter the dataset for labels 'J' and 'P'
df['label_JP'] = df['type'].apply(lambda x: 'J' if 'J' in x else ('P' if 'P' in x else np.nan))
data_JP = df.dropna(subset=['label_JP'])

# Text length analysis for labels 'J' and 'P'
data_JP['post_length'] = data_JP['full_preprocessed_text'].apply(lambda x: len(x.split()))

# Plot the distribution of post length by type
sns.histplot(data=data_JP, x='post_length', hue='label_JP', bins=50, kde=True)
plt.title('Distribution of Post Length by Type (J vs P)')
plt.xlabel('Post Length')
plt.ylabel('Frequency')
plt.show()


# %% code cell 42
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import spacy
!pip install
import nltk
nltk.download('punkt')

# Load dataset
data = pd.read_csv("/content/drive/MyDrive/mbti_1.csv")

# Download stopwords
nltk.download('stopwords')

# Load Spacy's English language model
nlp = spacy.load('en_core_web_sm')

# Combine NLTK and sklearn stopwords
stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))

# Filter posts related to 'N' and 'S' labels
n_posts = data[data['type'].str.contains('N')]['posts']
s_posts = data[data['type'].str.contains('S')]['posts']

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return " ".join(tokens)

# Preprocess the posts
data['cleaned_posts'] = data['posts'].apply(preprocess_text)
n_posts_cleaned = n_posts.apply(preprocess_text)
s_posts_cleaned = s_posts.apply(preprocess_text)

# Basic data inspection
print(data.info())
print(data.describe())

# Text length analysis
data['post_length'] = data['cleaned_posts'].apply(lambda x: len(x.split()))
sns.histplot(data=data, x='post_length', hue='type', bins=50, kde=True)
plt.title('Distribution of Post Length by Type')
plt.show()


# %% code cell 43


# %% code cell 44
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("/content/drive/MyDrive/mbti_1.csv")

# Combine NLTK and sklearn stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return " ".join(tokens)

# Preprocess the posts
data['cleaned_posts'] = data['posts'].apply(preprocess_text)

# Filter posts related to 'N' and 'S' labels
n_posts_cleaned = data[data['type'].str.contains('N')]['cleaned_posts']
s_posts_cleaned = data[data['type'].str.contains('S')]['cleaned_posts']

# Generate word clouds
def create_wordcloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.show()

# Create word clouds for 'N' and 'S' posts
create_wordcloud(n_posts_cleaned, 'Word Cloud for Intuitive (N) Posts')
create_wordcloud(s_posts_cleaned, 'Word Cloud for Sensing (S) Posts')


# %% code cell 45
# Filter posts related to 'N' and 'S' labels
ns_posts_cleaned = data[data['type'].str.contains('N|S')]['cleaned_posts']

# Combine all posts into a single string
combined_text_ns = " ".join(ns_posts_cleaned)

# Generate and display the word cloud
wordcloud_ns = WordCloud(width=800, height=400, background_color='white').generate(combined_text_ns)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_ns, interpolation='bilinear')
plt.title('Combined Word Cloud for Intuitive (N) and Sensing (S) Posts', fontsize=18)
plt.axis('off')
plt.show()


# %% [markdown] cell 46
# ### ***DL MODELS***
# %% code cell 47
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from torchtext.vocab import GloVe
!pip install sentence-transformers
from sentence_transformers import SentenceTransformer


# %% code cell 48
import pandas as pd
import json
import re
import numpy as np
import spacy
import tqdm
import xgboost as xgb
import lightgbm as lgb
import nltk
!pip install emoji
import emoji

!pip install catboost
from catboost import CatBoostClassifier # This is where CatBoostClassifier is defined

# Download NLTK data for tokenization
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier # Remove CatBoostClassifier from here
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from nltk import pos_tag, word_tokenize
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# %% code cell 49
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.fasttext import FastText
import transformers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout
from torchtext.vocab import GloVe
!pip install sentence-transformers
from sentence_transformers import SentenceTransformer


# %% code cell 50
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device

# %% code cell 51
# Preprocess text (simple preprocessing considering only removal of URLs and lowercasing)
data['posts'] = data['posts'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
data['posts'] = data['posts'].str.lower()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['posts'], data['class'], test_size=0.2, random_state=42)

# %% code cell 52
# Train-test split
X_train_ns, X_test_ns, y_train_ns, y_test_ns = train_test_split(X_ns, y_ns, test_size=0.2, random_state=42)

# %% code cell 53
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_ns)
X_train_seq = tokenizer.texts_to_sequences(X_train_ns)
X_test_seq = tokenizer.texts_to_sequences(X_test_ns)

# Pad sequences to ensure uniform input size
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)


# %% code cell 54
# Train a Word2Vec model
sentences = [sentence.split() for sentence in X_train_ns]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Create an embedding matrix
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = w2v_model.wv[word] if word in w2v_model.wv else None
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# %% code cell 55
# Load the GloVe model
glove = GloVe(name='6B', dim=100)

# Create an embedding matrix for Glove
embedding_matrix_glove = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove[word]
    if embedding_vector is not None:
        embedding_matrix_glove[i] = embedding_vector


# %% code cell 56
print(type(X_train_ns))
print(type(X_test_ns))
X_train_ns = X_train_ns.tolist()
X_test_ns = X_test_ns.tolist()


# %% code cell 57
# Initialize Sentence Transformer Model
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encode sentences (for simplification we use mean pooling of embeddings)
X_train_embeddings = sbert_model.encode(X_train_ns, show_progress_bar=True)
X_test_embeddings = sbert_model.encode(X_test_ns, show_progress_bar=True)

# %% code cell 58
def build_model(embedding_matrix, lstm_type='lstm'):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=100,
                        weights=[embedding_matrix], trainable=False))
    if lstm_type == 'lstm':
        model.add(LSTM(100))
    elif lstm_type == 'bilstm':
        model.add(Bidirectional(LSTM(100)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# %% code cell 59
# Check the data types of your target variable
print(y_train_ns.dtype)

# Check for string values in your features (after padding)
print(np.unique(X_train_pad))

# If you find string values, you need to convert them to numerical representations.
# For example, if the padding token is a string, you can convert it to an integer:

X_train_pad = np.where(X_train_pad == 'your_padding_token', 0, X_train_pad).astype(np.float32)
X_test_pad = np.where(X_test_pad == 'your_padding_token', 0, X_test_pad).astype(np.float32)

# If your target variable contains strings, you need to encode them numerically (e.g., using label encoding or one-hot encoding)

# %% code cell 60
# LSTM Model
model_w2v_lstm = build_model(embedding_matrix, 'lstm')
model_w2v_lstm.fit(X_train_pad, y_train_ns, epochs=35, batch_size=64, validation_data=(X_test_pad, y_test_ns))

# %% code cell 61
!pip install sklearn.metrics
from sklearn.metrics import classification_report

# %% code cell 62
# Predict probabilities for each class for the test data using the trained LSTM model
y_pred_probs_lstm = model_w2v_lstm.predict(X_test_pad)

# Convert probabilities to class labels
y_pred_lstm = (y_pred_probs_lstm > 0.5).astype(int)

# Generate and print the classification report
classification_report_lstm = classification_report(y_test, y_pred_lstm)
print("Classification Report for LSTM Model:")
print(classification_report_lstm)


# %% code cell 63
# Bi-LSTM Model
model_w2v_bilstm = build_model(embedding_matrix, 'bilstm')
model_w2v_bilstm.fit(X_train_pad, y_train, epochs=35, batch_size=64, validation_data=(X_test_pad, y_test))

# %% code cell 64
# Predict probabilities for each class for the test data using the trained Bi-LSTM model
y_pred_probs_bilstm = model_w2v_bilstm.predict(X_test_pad)

# Convert probabilities to class labels
y_pred_bilstm = (y_pred_probs_bilstm > 0.5).astype(int)

# Generate and print the classification report
classification_report_bilstm = classification_report(y_test, y_pred_bilstm)
print("Classification Report for Bi-LSTM Model:")
print(classification_report_bilstm)


# %% code cell 65

# CNN Model with GloVe
lstm_model_glove = build_lstm_model(embedding_matrix_glove)
lstm_model_glove.fit(X_train_pad, y_train, epochs=35, batch_size=64, validation_data=(X_test_pad, y_test))


# %% code cell 66
from sklearn.metrics import classification_report

# Predict probabilities for each class for the test data using the trained  model
y_pred_probs = lstm_model_glove.predict(X_test_pad)

# Convert probabilities to class labels
y_pred = (y_pred_probs > 0.5).astype(int)

# Generate and print the classification report
classification_report_lstm = classification_report(y_test, y_pred)
print("Classification Report for CNN Model with GloVe:")
print(classification_report_lstm)


# %% code cell 67
# lstm Model with Sentence Embeddings
lstm_model_sentence = build_lstm_model(None)  # No embedding layer needed
lstm_model_sentence.fit(X_train_embeddings, y_train, epochs=35, batch_size=64, validation_data=(X_test_embeddings, y_test))

# %% code cell 68


# %% [markdown] cell 69
# ### ***BERT***
# %% code cell 70


# %% code cell 71
tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased", max_length=512)

# %% code cell 72
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
model.to(device)

# %% code cell 73
SIZE= data.shape[0]

train_texts= list(data.posts[:SIZE//2])

val_texts=   list(data.posts[SIZE//2:(3*SIZE)//4 ])

test_texts=  list(data.posts[(3*SIZE)//4:])

train_labels= list(data.labels[:SIZE//2])

val_labels=   list(data.labels[SIZE//2:(3*SIZE)//4])

test_labels=  list(data.labels[(3*SIZE)//4:])

# %% code cell 74
len(train_texts), len(val_texts), len(test_texts)

# %% code cell 75
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# %% code cell 76
class DataLoader(Dataset):
    """
    Custom Dataset class for handling tokenized text data and corresponding labels.
    Inherits from torch.utils.data.Dataset.
    """
    def __init__(self, encodings, labels):
        """
        Initializes the DataLoader class with encodings and labels.

        Args:
            encodings (dict): A dictionary containing tokenized input text data
                              (e.g., 'input_ids', 'token_type_ids', 'attention_mask').
            labels (list): A list of integer labels for the input text data.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tokenized data and the corresponding label for a given index.

        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            item (dict): A dictionary containing the tokenized data and the corresponding label.
        """
        # Retrieve tokenized data for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the given index to the item dictionary
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            (int): The number of data items in the dataset.
        """
        return len(self.labels)

# %% code cell 77
train_dataloader = DataLoader(train_encodings, train_labels)

val_dataloader = DataLoader(val_encodings, val_labels)

test_dataset = DataLoader(test_encodings, test_labels)

# %% code cell 78
from transformers import TrainingArguments, Trainer

# %% code cell 79
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """
    Computes accuracy, F1, precision, and recall for a given set of predictions.

    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.

    Returns:
        dict: A dictionary containing the following metrics:
            - Accuracy (float): The proportion of correctly classified instances.
            - F1 (float): The macro F1 score, which is the harmonic mean of precision
              and recall. Macro averaging calculates the metric independently for
              each class and then takes the average.
            - Precision (float): The macro precision, which is the number of true
              positives divided by the sum of true positives and false positives.
            - Recall (float): The macro recall, which is the number of true positives
              divided by the sum of true positives and false negatives.
    """
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

# %% code cell 80
import os
os.listdir('../models_output/text_clf')
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.0
)

training_args = TrainingArguments(
    output_dir="./models_output/text_clf",
    do_train=True,
    do_eval=True,

    # Training settings
    num_train_epochs=100,                      # From hyperparameter table
    per_device_train_batch_size=16,            # Training batch size
    per_device_eval_batch_size=32,             # Validation batch size
    learning_rate=1e-5,                        # From table
    weight_decay=0.01,                         # From table
    warmup_steps=100,                          # From table
    fp16=True,                                 # Mixed precision

    # Logging and evaluation frequency
    logging_strategy="steps",
    logging_steps=100,                         # From table
    evaluation_strategy="steps",
    eval_steps=100,                            # From table

    # Saving strategy
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,               # Use best checkpoint

    # Early stopping requires a callback (added below)
    report_to="none",                          # Disable wandb if not needed
)


# %% code cell 82
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)


# %% code cell 83
trainer.train()

