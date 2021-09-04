from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from tfidf import tfidf_matrix_train, tfidf_matrix_test, train_df

y = train_df['Match Percentage']
#rf = RandomForestRegressor().fit(tfidf_matrix_train, y)

from sklearn.ensemble import GradientBoostingRegressor
'''gbr_reg = GradientBoostingRegressor(n_estimators=1000, 
                                    learning_rate=0.01, 
                                    max_depth=1, random_state=31).fit(tfidf_matrix_train, y)'''

import xgboost as XGB
xgb = XGB.XGBRegressor(learning_rate=0.01, 
                        n_estimators=1000, 
                        objective='reg:squarederror', 
                        random_state = 31).fit(tfidf_matrix_train, y)

def submission(model, test_sentences):
    test1 = pd.read_csv('dataset/test.csv')
    preds = model.predict(test_sentences)
    prediction = pd.DataFrame(preds, columns = ['Match Percentage'])
    sub_df = pd.concat([test1, prediction], axis = 1)
    return sub_df

sub = submission(xgb, tfidf_matrix_test)
sub.to_csv('submission file/Submission.csv')
print(sub.head())