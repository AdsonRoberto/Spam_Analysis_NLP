import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag, pos_tag_sents
import string

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("tagsets")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")

!python -m spacy download 'en_core_web_sm'

nlp = spacy.load('en_core_web_sm')

spam = pd.read_csv("td_atv_no.csv")
spam.head()

from google.colab import drive
drive.mount('/content/drive')

for message in spam["Message"]:
  print(message)
  break


tokenizada = []

for linha in spam['Message']:
  tokenizada.append(sent_tokenize(linha, language="english"))

spam['tokenizacao'] = tokenizada

spam.head()