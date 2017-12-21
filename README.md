# Samsung_sms_classification
code for one of the hackthon organised by t-hub and samsung (hack2innovate)

Samsung SMS Classification (ham,spam, info)

98.93 (accuracy) score on leaderboard. Approach is described below.

Step 1: Pre Processing of given dataset
   1.1  Given Dataset has too many punctuation and some unnessary punctuation in it so we use the regex 	
        for removing them
   1.2  We expand the various contraction used in the english language e.g isn't to is not or abt to about 	etc.
   1.3  We remove the stopwords and done the lemetization

Step 2: Tokenozation of given dataset using nltk

Step 3: Converting Token_corpus to word vectors using word2vec technique and use tf_idf_vectorizer from 	nltk to average the word vectors and reduse the span of it

Step 4: After that we use the randomforest classifier to classify the word_vectors made above
