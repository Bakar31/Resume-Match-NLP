# required libraries
import pandas as pd
from lightgbm import LGBMRegressor
import xgboost as XGB
from countvec import countvec_matrix_train, countvec_matrix_test, train_df

# dependent features
y = train_df['Match Percentage']

# XGBoost model
xgb = XGB.XGBRegressor(learning_rate=0.3,
                        n_estimators=200,
                        max_depth=7,
                        max_delta_step = 50,
                        random_state = 31).fit(countvec_matrix_train, y)

# turning to float for lgbm
countvec_matrix_train = countvec_matrix_train.astype('float32')
countvec_matrix_test = countvec_matrix_test.astype('float32')

# LightBGM model (Best Model)
lgbm = LGBMRegressor(num_leaves=31,
                    learning_rate = 0.2,
                    n_estimators = 200,
                    reg_alpha = 1,
                    random_state=31).fit(countvec_matrix_train, y)

# prediction and submission file creation
def submission(model, test_sentences):
    test1 = pd.read_csv('dataset/test.csv')
    preds = model.predict(test_sentences)
    prediction = pd.DataFrame(preds, columns = ['Match Percentage'])
    sub_df = pd.concat([test1, prediction], axis = 1)
    return sub_df

# predinting with xgboost model
sub = submission(xgb, countvec_matrix_test)
sub.to_csv('submission file/lgbm countvec best.csv')
print(sub.head())