from features import train_df, test_df

train_df.drop('resumes', axis = 1, inplace=True)
test_df.drop('resumes', axis = 1, inplace=True)

x = train_df.drop('Match Percentage', axis = 1)
y = train_df['Match Percentage']

from sklearn.ensemble import GradientBoostingRegressor
gbr_reg = GradientBoostingRegressor(n_estimators=1000, 
                                    learning_rate=0.01, 
                                    max_depth=1, 
                                    random_state=31).fit(x, y)
