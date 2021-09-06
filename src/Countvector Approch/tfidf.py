# required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from processing import train_lemma, test_lemma

# datasets
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# preprocessed and final dataset dataset
train_df = pd.concat([train, pd.DataFrame(train_lemma, columns=['resumes'])], axis = 1)
test_df = pd.concat([test, pd.DataFrame(test_lemma, columns=['resumes'])], axis = 1)

print(train_df.head())
print(test_df.head())

# apply TFIDF
tfidf = TfidfVectorizer(max_features=10000, 
                        strip_accents='unicode', 
                        analyzer='word',
                        lowercase=False,
                        ngram_range=(1, 1), 
                        stop_words = 'english')

tfidf_matrix_train = tfidf.fit_transform(train_df['resumes'])
tfidf_matrix_test = tfidf.transform(test_df['resumes'])
print(tfidf_matrix_train.shape)
print(tfidf_matrix_test.shape)