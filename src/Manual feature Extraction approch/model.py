from features import train_df, test_df

train_df.drop('resumes', axis = 1, inplace=True)
test_df.drop('resumes', axis = 1, inplace=True)

x = train_df.drop('Match Percentage', axis = 1)
y = train_df['Match Percentage']

'''from sklearn.ensemble import GradientBoostingRegressor
gbr_reg = GradientBoostingRegressor().fit(x, y)'''

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor().fit(x, y)
