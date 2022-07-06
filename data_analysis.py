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

coluna_minusculo = []
coluna_tokenizada = []
coluna_sem_stop_word = []
coluna_sem_tag = []

stop_word = stopwords.words("english")

for linha in spam['Message']:
  #Texto minisculo
  texto_minusculo = linha.lower()
  coluna_minusculo.append(texto_minusculo)

  #Removendo tags html
  texto_sem_tag = re.sub(r'<.*?>', ' ', texto_minusculo)
  coluna_sem_tag.append(texto_sem_tag)

  #Tokenizando texto em palavras
  texto_tokenizado = word_tokenize(texto_sem_tag, language="english")
  coluna_tokenizada.append(texto_tokenizado)

  #Sem stopword
  texto_sem_sw = [w for w in texto_tokenizado if not w in stop_word]
  coluna_sem_stop_word.append(texto_sem_sw)

spam['minusculo'] = coluna_minusculo
spam['sem_tag'] = coluna_sem_tag
spam['tokenizacao'] = coluna_tokenizada
spam['sem_stop_word'] = coluna_sem_stop_word

type(spam["Message"][0][0])

