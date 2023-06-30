#!/usr/bin/env python
# coding: utf-8

# # House Prices prediction Advanced Regression Techniques

# #Following tasks to be performed:

# Task 1: Data Understanding: Begin by exploring the dataset and understanding its structure, including the meaning and type of each feature.
# 
# Task 2: Data Cleaning: Perform data cleaning tasks to handle missing values, outliers, and inconsistencies in the dataset.
# 
# Task 3: Feature Engineering: Perform feature engineering to enhance the predictive power of the dataset. This may include creating new features, transforming existing features, or selecting relevant features.
# 
# Task 4: Data Preprocessing: Prepare the cleaned dataset for model training. This involves scaling numerical features, encoding categorical variables, and splitting the data into training and testing sets.
# 
# Task 5: Model Training and Evaluation: Choose an appropriate regression model (e.g., linear regression, random forest, or gradient boosting) and train it on the preprocessed dataset. Evaluate the model's performance using suitable metrics like mean squared error (MSE) or root mean squared error (RMSE).
# 
# Task 6: Model Optimization: Fine-tune the hyperparameters of the chosen model to improve its performance. You can use techniques like cross-validation or grid search to find the best parameter values.
# 
# Task 7: Model Deployment: Once you have a satisfactory model, deploy it to make predictions on new, unseen data. You can use the trained model to predict house prices for new instances and assess its real-world applicability.
# 
# Task 8: Linkedin Post: Once you complete all the above tasks, make a linkedin post from your account for the entire Final Assignment completion.
# 

# In[1]:


#Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
 
from scipy import stats
from scipy.stats import norm, skew


# In[2]:


import warnings
warnings.simplefilter("ignore")


# In[3]:


#Loading the data from csv file to pandas dataframe
df = pd.read_csv(r'D:\DIG\house prediction assignment 7&8 unmessenger.csv')


# In[4]:


df


# In[5]:


## Shape of dataframe
df.shape


# In[6]:


## Information about the dataset
df.info()


# In[7]:


## Lets see the discription of the dataset
df.describe().T


# # Data Preprocessing & Cleaning

# In[8]:


# Converting dtypes of columns
df['date'] = pd.to_datetime(df['date'])
df['price']     = df['price'].astype('int64')
df['bedrooms']  = df['bedrooms'].astype('int64')
df['bathrooms'] = df['bathrooms'].astype('int64')
df['floors']    = df['floors'].astype('int64')
df['street']    = df['street'].astype('string')
df['city']      = df['city'].astype('string')
df['statezip']  = df['statezip'].astype('string')
df['country']   = df['country'].astype('string')


# In[9]:


# Extract features from the datetime column
df.insert(1, "year", df.date.dt.year)


# In[10]:


df


# # Handling missing, duplicate and 0 values

# In[11]:


# Check for missing or null values
df.isnull().sum()


# In[12]:


# Removing duplicate rows from the dataframe if there are in the data
df.drop_duplicates()
df.shape


# In[13]:


#There are no duplicate values in the dataframe.


# In[14]:


# Removing rows having price values 0

# Checking price having 0 values
price_zero = (df.price == 0).sum()
print(price_zero)

# drop the column having price value 0
df['price'].replace(0, np.nan, inplace = True)
df.dropna(inplace=True)

# Checking shape of the dataset
print(df.shape)


# In[15]:


# Dropping unnecessary columns from the dataset
df = df.drop(['date', 'street'], axis = 1)
df.head(5)


# In[16]:


# Number of unique value counts in the dataset
df.nunique(axis = 0)


# In[17]:


df.dtypes


# In[18]:


# Treating 'statezip' column and extracting the numeric code only
df['statezip'] = df['statezip'].str.split().str[1]

# Reshape the column to a 2D array with a single feature
df['statezip'] = np.reshape(df['statezip'].values, (-1, 1))


# In[19]:


df['statezip'] = df['statezip'].astype('int64')
df['floors']    = df['floors'].astype('int64')


# In[20]:


df.shape


# In[21]:


df


# In[22]:


## Univariate & Bivariant analysis of categorical columns
plt.figure(figsize=(15, 30))

plt.subplot(6, 2, 1)
pd.value_counts(df['bedrooms']).plot(kind='bar')
plt.title("Bedroom Counts")

plt.subplot(6, 2, 2)
sns.barplot(x = df['bedrooms'], y = df.price)
plt.title("Bedroom - Price")

plt.subplot(6, 2, 3)
pd.value_counts(df['bathrooms']).plot(kind='bar')
plt.title("Bathroom Counts")

plt.subplot(6, 2, 4)
sns.barplot(x = df['bathrooms'], y = df.price)
plt.title("Bathroom - Price")

plt.subplot(6, 2, 5)
pd.value_counts(df['floors']).plot(kind='bar')
plt.title("Floor Counts")

plt.subplot(6, 2, 6)
sns.barplot(x = df['floors'], y = df.price)
plt.title("Floor - Price")

plt.subplot(6, 2, 7)
pd.value_counts(df['waterfront']).plot(kind='bar')
plt.title("Waterfront Counts")

plt.subplot(6, 2, 8)
sns.barplot(x = df['waterfront'], y = df.price)
plt.title("waterfront - Price")

plt.subplot(6, 2, 9)
pd.value_counts(df['view']).plot(kind='bar')
plt.title("View Counts")

plt.subplot(6, 2, 10)
sns.barplot(x = df['view'], y = df.price)
plt.title("View - Price")

plt.subplot(6, 2, 11)
pd.value_counts(df['condition']).plot(kind='bar')
plt.title("Condition Counts")

plt.subplot(6, 2, 12)
sns.barplot(x = df['condition'], y = df.price)
plt.title("Condition - Price")

plt.show()


# In[23]:


# Checking distribution of price
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15,4)

sns.distplot(df['price'],color="#B31312",kde=True)
plt.show()


# In[24]:


# Lets visualize the outliers using box-plot
sns.boxplot(df['price'])
plt.title('Outliers present in the Data')
plt.show()


# In[25]:


## Calculate the first quartile (Q1) and third quartile (Q3)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

## Calculate the interquartile range (IQR)
IQR = Q3 - Q1

## Define the lower and upper bounds for outlier detection
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

## Find the outliers in the column
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

## Count the number of outliers
outlier_count = len(outliers)

## Print the number of outliers
print("Number of outliers in 'price' column :",outlier_count)
outliers.head()


# In[26]:


# Convert the outliers to NaN
df['price'][outliers.index] = np.nan
df.head(3)


# In[27]:


df['price'].isnull().sum()


# In[28]:


# Fill the NaN values with the mean

# Calculate the mean value (rounded to 0 decimal places)
mean_value = round(df['price'].mean())

# Fill null values with the rounded mean value
df['price'].fillna(mean_value, inplace=True)
#df['price'] = df['price'].fillna(df['price'].mean())
df.head(3)


# In[29]:


# Again checking distribution of price
sns.distplot(df['price'],color="#2E4F4F",kde=True)
plt.show()


# # Task 3: Feature Engineering
# 
# A. Log Transformation on Target Variable
# With the help of Q-Q plot we see whether the target variable is Normally Distributed or not, as Linear mostly like Normally Distributed Data.

# In[30]:


# Plotting QQ-plot
fig = plt.figure()
res = stats.probplot(df['price'], plot=plt)
plt.show()


# As the target variable (price) is very skewed, so we apply log-transformation on target varibale to make it Normally Distributed.

# In[31]:


# Applying log-transformation
df['price'] = np.log(df['price'])

## Again plotting QQ-plot
fig = plt.figure()
res = stats.probplot(df['price'], plot=plt)
plt.show()


# In[32]:


# Checking distribution of price again
sns.distplot(df['price'],color="#FF8400",kde=True,fit=norm)


# In[33]:


# Creating heatmap to check the correlation in the dataset
plt.rcParams['figure.figsize'] = (12,12)

sns.heatmap(df.corr(), annot=True)
plt.title('Heat Map', size=20)
plt.yticks(rotation = 0)
plt.show()


# Encoding Independent Variables

# In[34]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to the 'city' column
df['city'] = label_encoder.fit_transform(df['city'])
df['country'] = label_encoder.fit_transform(df['country'])
df['bedrooms'] = label_encoder.fit_transform(df['bedrooms'])
df['bathrooms'] = label_encoder.fit_transform(df['bathrooms'])
df['price'] = label_encoder.fit_transform(df['price'])
df['sqft_living'] = label_encoder.fit_transform(df['sqft_living'])
df['sqft_lot'] = label_encoder.fit_transform(df['sqft_lot'])
df['sqft_above'] = label_encoder.fit_transform(df['sqft_above'])
df['sqft_basement'] = label_encoder.fit_transform(df['sqft_basement'])
df['yr_built'] = label_encoder.fit_transform(df['yr_built'])
df['yr_renovated'] = label_encoder.fit_transform(df['yr_renovated'])


# In[35]:


df.columns


# In[36]:


# Creating heatmap to check the correlation in the dataset
plt.rcParams['figure.figsize'] = (12,12)
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.title('Heat Map', size=20)
plt.yticks(rotation = 0)
plt.show()


# In[37]:


df.columns


# In[38]:


data= df.copy()
data.head(3)


# In[39]:


data.dtypes


# # Task 5: Model Training and Evaluation
# 
# A. Splitting the Data and Target

# In[40]:


# Creating dependent and independent sets
X = data.drop(['price',], axis = 1)
Y = data['price']

print(X.head())
print(Y.head())


# In[41]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# # Feature Scaling

# In[42]:


# Perform standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler


# In[43]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[44]:


### Creating Models using diffenet algorithms

## 1. Creating a  Linear Regression Model
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(X_train, Y_train)
Y_pred1 = LR.predict(X_test)

## 2. Creating a Random Forest Model
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
Y_pred2 = RF.predict(X_test)

## 3. Creating a Gradient Boosting Model
from sklearn.ensemble import GradientBoostingRegressor

GB = GradientBoostingRegressor()
GB.fit(X_train, Y_train)
Y_pred3 = GB.predict(X_test)

## 4. Creating a SVR Model
from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, Y_train)
Y_pred4 = svr.predict(X_test)

## 5. Creating a Decision Tree Regressor Model
from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor()
DT.fit(X_train, Y_train)
Y_pred5 = DT.predict(X_test)

## 6. Creating Ridge Model
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, Y_train)
Y_pred6 = ridge.predict(X_test)


# Evaluation Metrics,
# R2 Score,
# Root Mean Squared Error (RMSE),
# Mean Squared Error (MSE)

# In[45]:


# Checking Model accuracy
from sklearn.metrics import r2_score, mean_squared_error

r2_score1 = r2_score(Y_test, Y_pred1)
r2_score2 = r2_score(Y_test, Y_pred2)
r2_score3 = r2_score(Y_test, Y_pred3)
r2_score4 = r2_score(Y_test, Y_pred4)
r2_score5 = r2_score(Y_test, Y_pred5)
r2_score6 = r2_score(Y_test, Y_pred6)

mse1 = mean_squared_error(Y_test, Y_pred1)
mse2 = mean_squared_error(Y_test, Y_pred2)
mse3 = mean_squared_error(Y_test, Y_pred3)
mse4 = mean_squared_error(Y_test, Y_pred4)
mse5 = mean_squared_error(Y_test, Y_pred5)
mse6 = mean_squared_error(Y_test, Y_pred6)

rmse1 = np.sqrt(mse1)
rmse2 = np.sqrt(mse2)
rmse3 = np.sqrt(mse3)
rmse4 = np.sqrt(mse4)
rmse5 = np.sqrt(mse5)
rmse6 = np.sqrt(mse6)

print("Linear Regression R2 Score :", r2_score1)
print("Linear Regression MSE :", mse1)
print("Linear Regression RMSE :", rmse1)
print("Random Forest R2 Score :", r2_score2)
print("Random Forest MSE :", mse2)
print("Random Forest RMSE :", rmse2)
print("Gradient Boosting R2 Score :", r2_score3)
print("Gradient Boosting MSE :", mse3)
print("Gradient Boosting RMSE :", rmse3)
print("SVR R2 Score :", r2_score4)
print("SVR MSE :", mse4)
print("SVR RMSE :", rmse4)
print("Decision Tree R2 Score :", r2_score5)
print("Decision Tree MSE :", mse5)
print("Decision Tree RMSE :", rmse5)
print("Ridge R2 Score :", r2_score6)
print("Ridge MSE :", mse6)
print("Ridge RMSE :", rmse6)


# In[46]:


# Checking scores of the models
print(LR.score(X_test,Y_test),": Linear Regression")
print(RF.score(X_test,Y_test),": Random Forest")
print(GB.score(X_test,Y_test),": Gradient Boosting")
print(svr.score(X_test,Y_test),": SVR")
print(DT.score(X_test,Y_test),": Decision Tree")
print(ridge.score(X_test,Y_test),": Ridge")


# # Model Optimization

# Evaluating the Algorithms

# In[47]:


# Creating dataframe for Models with scores
final_data = pd.DataFrame({'Models':['Linear Regression', 'Random Forest Regressor', 
                'Gradient Boosting Regressor', 'SVR', 'DecisionTreeRegressor', 'ElasticNet'], 
                        'R2_Score': [r2_score1, r2_score2, r2_score3, r2_score4, r2_score5, r2_score6]})

models_df = pd.DataFrame(final_data)

# Sort the DataFrame based on R2_Score in descending order
models_df_sorted = models_df.sort_values(by='R2_Score', ascending=False)

# Apply background gradient to the R2_Score column
models_df_sorted_styled = models_df_sorted.style.background_gradient(subset=['R2_Score'], cmap='Blues')
models_df_sorted_styled


# In[48]:


# Visualize the scores on barplot
plt.style.use('seaborn')
plt.figure(figsize = (10, 5))
sns.barplot(final_data['Models'],final_data['R2_Score'])

# Set the axis labels and title
plt.xlabel('Models', fontsize= 14)
plt.ylabel('R2 Scores', fontsize= 14)
plt.title('Comparison of R2 Scores', fontsize = 18)
plt.xticks(fontsize= 12, rotation = 90)
plt.yticks(fontsize= 12, rotation = 45)
plt.show()


# # Model Deployment Steps

# Creating Pickle File

# In[49]:


import pickle 
pickle.dump(RF,open("model_rf.pkl", 'wb'))


# Save processed data as a CSV file

# In[50]:


data.to_csv('processed_data.csv', index=False)

