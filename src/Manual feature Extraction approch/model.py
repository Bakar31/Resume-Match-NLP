import xgboost as XGB
from features import train_df, test_df

train_df.drop('resumes', axis = 1, inplace=True)
test_df.drop('resumes', axis = 1, inplace=True)

x = train_df.drop('Match Percentage', axis = 1)
y = train_df['Match Percentage']
print(x.head())

# xgboost regressior
xgb = XGB.XGBRegressor(learning_rate=0.005, 
                        n_estimators=700, 
                        objective='reg:squarederror', 
                        max_depth=8, 
                        reg_lambda = 1.3,
                        gamma = 1,
                        min_child_weight =1.5,
                        max_delta_step = 100,
                        random_state = 31).fit(x, y)
