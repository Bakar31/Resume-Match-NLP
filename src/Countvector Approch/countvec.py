# required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from processing import train_lemma, test_lemma

# datasets
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# preprocessed and final dataset dataset
train_df = pd.concat([train, pd.DataFrame(train_lemma, columns=['resumes'])], axis = 1)
test_df = pd.concat([test, pd.DataFrame(test_lemma, columns=['resumes'])], axis = 1)

print(train_df.head())
print(test_df.head())

# apply CountVectorizer
countvec = CountVectorizer(analyzer='word',
                        ngram_range=(1, 1), 
                        stop_words = 'english')

countvec_matrix_train = countvec.fit_transform(train_df['resumes'])
countvec_matrix_test = countvec.transform(test_df['resumes'])
print(countvec_matrix_train.shape)
print(countvec_matrix_test.shape)