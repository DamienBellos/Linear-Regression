import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

customers = pd.read_csv('Ecommerce Customers')
customers.head()
customers.info()
print(customers.describe())

# Create a jointplot to compare the Time on Website and Yearly Amount Spent
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

customers.columns
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers)


# Create a 2D hex bin plot comparing Time on App and Length of Membership.
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers, kind="hex")

# Compare relationships across the entire data set.
sns.pairplot(customers)

# Plot Yearly Amount Spent vs. Length of Membership. **
sns.lmplot(x='Yearly Amount Spent', y='Length of Membership', data=customers)

# Train and Test Data
customers.columns
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Train the Model
lm = LinearRegression()
lm.fit(X_train, y_train)


# coefficients of the model
print('Coefficients \n', lm.coef_)

# ## Predicting Test Data
predictions = lm.predict(X_test)

# ** Create a scatterplot of the real test values versus the predicted values. **
plt.scatter(y_test, predictions)
plt.xlabel('Y Values')
plt.ylabel('Y Predicted Values')


# ## Evaluating the Model
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RSME:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
metrics.explained_variance_score(y_test, predictions)

# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**
sns.distplot((y_test - predictions))


# ## Conclusion
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df
