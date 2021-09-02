from features import train_df, test_df
import pandas as pd

train_df.drop('resumes', axis = 1, inplace=True)
test_df.drop('resumes', axis = 1, inplace=True)

x = train_df.drop('Match Percentage', axis = 1)
y = train_df['Match Percentage']

'''from sklearn.ensemble import GradientBoostingRegressor
gbr_reg = GradientBoostingRegressor(n_estimators=1000, 
                                    learning_rate=0.01, 
                                    max_depth=1, 
                                    random_state=31).fit(x, y)'''



import xgboost as XGB
xgb = XGB.XGBRegressor(learning_rate=0.01, 
                        n_estimators=100, 
                        random_state = 31).fit(x, y)

def submission(model, test):
    test1 = pd.read_csv('dataset/test.csv')
    preds = model.predict(test)
    prediction = pd.DataFrame(preds, columns = ['Match Percentage'])
    sub_df = pd.concat([test1, prediction], axis = 1)
    return sub_df

sub = submission(xgb, test_df)
sub.to_csv('submission file/Submission-5.csv')
print(sub.head())