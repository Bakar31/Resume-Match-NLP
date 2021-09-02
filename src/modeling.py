import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from preparing_to_train import train_df, test_df

max_vocab_length = 10000
max_length = 100

x = train_df["resumes"].to_numpy()
y = train_df["Match Percentage"].to_numpy()
test_sentences = test_df["resumes"].to_numpy()

text_vectorizer = TextVectorization(max_tokens=None,
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

model.fit(x, y)