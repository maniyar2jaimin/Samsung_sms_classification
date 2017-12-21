import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from feature_extractors import bow_extractor, tfidf_extractor
from feature_extractors import averaged_word_vectorizer

def prepare_datasets(corpus, labels, test_data_proportion=0.15):
    train_X, test_X, train_Y, test_Y = train_test_split(corpus, labels,
                                                        test_size=0.15, random_state=69)
    return train_X, test_X, train_Y, test_Y

train_data = pd.read_csv("/home/ranjitanair13ranju/TRAIN_SMS (1).csv",encoding="ISO-8859-1")
print(train_data.head())
raw_data = train_data[["Message"]].values.tolist()
raw_label = train_data[["Label"]].values.tolist()

raw_data_1 = []
for raw in raw_data:
    text = str(raw).strip("['\()]")
    raw_data_1.append(text)

raw_label_1 = []
for lab in raw_label:
    text = str(lab).strip("['\()]")
    raw_label_1.append(text)


print(raw_data_1[0])
print(raw_label_1[0])

train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(raw_data_1,raw_label_1)

print(train_corpus[0])
print(train_labels[0])

from normalization import normalize_corpus

norm_train_corpus = normalize_corpus(train_corpus)
norm_test_corpus = normalize_corpus(test_corpus)
from feature_extractors import averaged_word_vectorizer
from feature_extractors import tfidf_weighted_averaged_word_vectorizer
import nltk
import gensim

tokenized_train = [nltk.word_tokenize(text)
                   for text in norm_train_corpus]

tokenized_test = [nltk.word_tokenize(text)
                   for text in norm_test_corpus]

# build word2vec model
model = gensim.models.Word2Vec(tokenized_train,
                               size=700,
                               window=200,
                               min_count=30,
                               sample=1e-3)


# tfidf weighted averaged word vector features
from feature_extractors import tfidf_weighted_averaged_word_vectorizer
# tfidf features
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

vocab = tfidf_vectorizer.vocabulary_
tfidf_wv_train_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train,
                                                                  tfidf_vectors=tfidf_train_features,
                                                                  tfidf_vocabulary=vocab,
                                                                  model=model,
                                                                  num_features=700)

tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test,
                                                                 tfidf_vectors=tfidf_test_features,
                                                                 tfidf_vocabulary=vocab,
                                                                 model=model,
                                                                 num_features=700)

from sklearn import metrics
import numpy as np

def get_metrics(true_labels, predicted_labels):

    print 'Accuracy:', np.round(
                        metrics.accuracy_score(true_labels,
                                               predicted_labels),
                        4)
    print 'Precision:', np.round(
                        metrics.precision_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        4)
    print 'Recall:', np.round(
                        metrics.recall_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        4)
    print 'F1 Score:', np.round(
                        metrics.f1_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        4)

def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions,classifier


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier,LogisticRegression
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from mlxtend.classifier import StackingClassifier
from sklearn.preprocessing import LabelEncoder

mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge', n_iter=800)
rndf = RandomForestClassifier(max_depth=15, n_estimators=300,oob_score=True,n_jobs=6,verbose=1)
qda = QuadraticDiscriminantAnalysis()
lr = LogisticRegression()
#clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

sclf = StackingClassifier(classifiers=[svm,qda,rndf],
                          meta_classifier=lr)
#eclf1 = VotingClassifier(estimators=[('rf', rndf), ('qda', qda)], voting='soft')
#for clf, label in zip([sclf],
#                      ['sclf']):
#    le = LabelEncoder ()
#    train_labels = le.fit(train_labels)
#    scores = cross_val_score(clf, tfidf_wv_train_features, train_labels,
#                                              cv=10, scoring='accuracy',n_jobs=6,verbose=1)
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
#          % (scores.mean(), scores.std(), label))

prediction_set,classifier = train_predict_evaluate_model(rndf,tfidf_wv_train_features,train_labels,tfidf_wv_test_features,test_labels)


test_data = pd.read_csv("TEST_SMS.csv",encoding="ISO-8859-1")
print(test_data.head())
raw_test_data = test_data[["Message"]].values.tolist()

raw_test_data_1 = []
for raw in raw_test_data:
    text = str(raw).strip("['\()]")
    raw_test_data_1.append(text)

norm_actual_test_corpus = normalize_corpus(raw_test_data_1)

tokenized_actual_test = [nltk.word_tokenize(text)
                   for text in norm_actual_test_corpus]

tfidf_actual_test_features = tfidf_vectorizer.transform(norm_actual_test_corpus)

actual_test_feature = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_actual_test,
                                tfidf_vectors=tfidf_actual_test_features,
                                tfidf_vocabulary=vocab,
                                model=model,
                           num_features=700)


dev_prediction_set = classifier.predict(actual_test_feature)

columns = ["Message"]
df = pd.DataFrame(data=dev_prediction_set,columns=columns)
df.to_csv("final_submit_sms.csv",columns=columns,index=True)
