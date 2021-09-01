import pandas as pd
from modeling import model, test_sentences

def submission(model, test_sentences = test_sentences):
    test1 = pd.read_csv('dataset/test.csv')
    preds = model.predict(test_sentences)
    prediction = pd.DataFrame(preds, columns = ['Match Percentage'])
    sub_df = pd.concat([test1, prediction], axis = 1)
    return sub_df

sub = submission(model)
sub.to_csv('Submission-1.csv')
print(sub.head())