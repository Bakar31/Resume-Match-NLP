import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from preparing_to_train import train_df, test_df

max_vocab_length = 10000
max_length = 100

'''x = train_df["resumes"].to_numpy()
y = train_df["Match Percentage"].to_numpy()
test_sentences = test_df["resumes"].to_numpy()'''

requiredText = train_df["resumes"].values
requiredTarget = train_df["Match Percentage"].values
requiredText_test = test_df["resumes"].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)
WordFeatures_test = word_vectorizer.transform(requiredText_test)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.12)
print(X_train.shape)
print(X_test.shape)

from sklearn.ensemble import GradientBoostingRegressor
gbr_reg = GradientBoostingRegressor(n_estimators=1000, 
                                    learning_rate=0.01, 
                                    max_depth=1, 
                                    random_state=31).fit(X_train, y_train)


print(gbr_reg.score(X_test, y_test))


import pandas as pd

def submission(model, test_sentences):
    test1 = pd.read_csv('dataset/test.csv')
    preds = model.predict(test_sentences)
    prediction = pd.DataFrame(preds, columns = ['Match Percentage'])
    sub_df = pd.concat([test1, prediction], axis = 1)
    return sub_df

sub = submission(gbr_reg, WordFeatures_test)
sub.to_csv('submission file/Submission-20.csv')
print(sub.head())




'''text_vectorizer = TextVectorization(max_tokens=None,
                                    standardize="lower_and_strip_punctuation",
                                    split="whitespace", 
                                    ngrams=None,
                                    output_mode="int",
                                    output_sequence_length=None)



text_vectorizer = TextVectorization(max_tokens=max_vocab_length,
                                    output_mode="int",
                                    output_sequence_length=max_length)
    
text_vectorizer.adapt(x)
text_vectorizer.adapt(test_sentences)

model = Pipeline([
                    ("tfidf", TfidfVectorizer()), 
                    ("reg", RandomForestRegressor())
])

model.fit(x, y)'''