


import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_columns = None

# load data from csv file 
df = pd.read_csv("housing.csv")
# show first 5 rows of data
df.head()
# show info about data types and null values
df.info()
# show summary statistics of data ocean_proximity
df['ocean_proximity']
# change to category
df = df.astype({'ocean_proximity': 'category'})
# show info about data types and null values
df.info()

# show data with median_house_value > 300000
df[df['median_house_value']>300000]

df['rooms_per_household'] = df['total_rooms']/df['households']
df.info()
df['rooms_per_household']

df = df.drop('rooms_per_household', axis=1)

#  gives summary statistics
df.describe()

# show distribution of data in histogram graph
df.hist(bins=50, figsize=(20,15))
plt.show()

# show distribution of ocean_proximity
df['ocean_proximity'].value_counts().plot(kind='bar')

# for splitting data into train and test
from sklearn.model_selection import train_test_split 


# split data into train and test
# test_size=0.1 means 10% of data will be used for testing
# random_state=42 means data will be split in the same way every time
train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)

train_set.describe()
test_set.describe()

# we take the median_house_value in the train set and put it in a variable
# named price_train and drop it from the train set
price_train = train_set['median_house_value']
train_set = train_set.drop(['median_house_value'], axis=1)

# we take the median_house_value in the test set and put it in a variable
# named price_test and drop it from the test set
price_test = test_set['median_house_value']
test_set = test_set.drop(['median_house_value'], axis=1)

# clean the data
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df.info()

# select the data that have category type
cat_attribs_df = train_set.select_dtypes(include=['category'])
cat_attribs_df
# put the data in a list 
cat_attribs = list(cat_attribs_df)
cat_attribs

# select the data that have numerical type int64 and float64
num_attribs_df = train_set.select_dtypes(include=['int64', 'float64'])
num_attribs_df
# put the data in a list
num_attribs = list(num_attribs_df)
num_attribs

# take the len of the num_attribs and cat_attribs
# len(num_attribs) + len(cat_attribs) and add one to another
# the sum of the both length of the two lists
len(num_attribs) + len(cat_attribs)
#the same output the sum of the two lists
len(list(train_set))


# when we have a house that have 0 bedrooms
num_pipeline = Pipeline([
    # we impute the missing values with the median
    # so it mean we fill the missing values with the median an aproximate value
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

# we apply the numerical pipeline to the num_attribs
# we apply the categorical pipeline to the cat_attribs
# the full_pipeline take both the numerical and categorical pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    # for the categorical data we use the OneHotEncoder
    # so we have in category three option 1H ocean , inland and near_bay
    # so we have 3 columns one for each option : 1H ocean , inland and near_bay
    # if the house is near_bay then the column will be 1 and  inland and 1h ocean will be 0
    ("cat", OneHotEncoder(), cat_attribs),
])

# this help us apply the full_pipeline to the train_set
# so the data will be prepared for the model to train
train_set_prepared = full_pipeline.fit_transform(train_set)
# we show the first row
train_set_prepared[0]
# we show the tenth row
train_set_prepared[10]

# we add linear regression to train the model
from sklearn.linear_model import LinearRegression

# we train the model
lin_reg = LinearRegression()
# we fit the model with the train_set_prepared and price_train
# .fit train the model
lin_reg.fit(train_set_prepared, price_train)

# we take the first 5 rows of data from the test set
# and put it in a variable named part_data_test
part_data_test = test_set.iloc[:5]
# we take the first 5 rows of data from the price_test
# and put it in a variable named part_price_test
part_price_test = price_test.iloc[:5]

# we apply the full_pipeline to the part_data_test
# so the data will be prepared for the model to train
data_test_prepared = full_pipeline.transform(part_data_test)

# we show the predictions of the model
# .predict predict the model
print("predictions: ", lin_reg.predict(data_test_prepared))
print("labels" , list(part_price_test))

# check the accuracy of the model by using mean_squared_error
from sklearn.metrics import mean_squared_error

# we apply the full_pipeline to the test_set
# so the data will be prepared for the model to train
test_set_prepared = full_pipeline.transform(test_set)
# we show the predictions of the model
# .predict predict the model
predictions = lin_reg.predict(test_set_prepared)

mse = mean_squared_error(price_test, predictions)
mse

rmse = np.sqrt(mse)
rmse

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

forest_reg.fit(train_set_prepared, price_train)

test_set_prepared = full_pipeline.transform(test_set)
predictions = forest_reg.predict(test_set_prepared)

mse = mean_squared_error(price_test, predictions)
mse

rmse = np.sqrt(mse)
rmse













