import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import spacy
from nltk.corpus import stopwords

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF

from xgboost import XGBClassifier

train_data = pd.read_csv("./data/train_sms.csv",encoding="ISO-8859-1")
print(train_data.head())

raw_data = train_data[["Message"]].values.tolist()

#nlp = spacy.load('en_vectors_web_lg')
stop_words = set(stopwords.words('english'))
special_char_list = ['-','_',',','.','[',']','{','}','<','>',':','@','#','$','%','^','&','*','(',')','+','=','...','..','....',';',':)']

word_vecs = []
#for raw in raw_data[567]:
#	tokens = nlp(str(raw).strip('"\'"\'[]'))
#	word_vec_temp = []
#	for token in tokens:
		#if(token.text not in stop_words and token.text not in special_char_list):
#		if(token.text not in stop_words):
#			print(token.text)
#			word_vec_temp.append(token.vector_norm)
#	word_vecs.append(word_vec_temp)


#print(word_vecs[0])

#word_array = np.asarray(word_vecs)

for
