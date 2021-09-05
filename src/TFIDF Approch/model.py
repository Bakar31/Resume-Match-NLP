# required libraries
import pandas as pd
from lightgbm import LGBMRegressor
import xgboost as XGB
from tfidf import tfidf_matrix_train, tfidf_matrix_test, train_df

# dependent features
y = train_df['Match Percentage']

# XGBoost model
xgb = XGB.XGBRegressor(learning_rate=0.005, 
                        n_estimators=700, 
                        objective='reg:squarederror', 
                        max_depth=8, 
                        reg_lambda = 1.3,
                        gamma = 1,
                        min_child_weight =1.5,
                        max_delta_step = 100,
                        random_state = 31).fit(tfidf_matrix_train, y)


# LightBGM model
lgbm = LGBMRegressor(num_leaves=31,
                    learning_rate = 0.01,
                    n_estimators = 1000,
                    reg_lambda = 2.5,
                    reg_alpha = 2,
                    random_state=31).fit(tfidf_matrix_train, y)

# prediction and submission file creation
def submission(model, test_sentences):
    test1 = pd.read_csv('dataset/test.csv')
    preds = model.predict(test_sentences)
    prediction = pd.DataFrame(preds, columns = ['Match Percentage'])
    sub_df = pd.concat([test1, prediction], axis = 1)
    return sub_df

# predinting with xgboost model
sub = submission(xgb, tfidf_matrix_test)
sub.to_csv('submission file/xgb sub.csv')
print(sub.head())