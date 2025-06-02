import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

df=pd.read_csv("real time data.csv")
print(df)
print(df.isnull().sum())
print(df.describe())
print(df.info())
print(df.columns)
print(df.shape)
print(df.head())
df=df.dropna()

df=df.drop_duplicates()

df.to_csv("cleaned_data.csv",index=False)
df=pd.read_csv("cleaned_data.csv")

print("Missing values after cleaning:")
print(df.isnull().sum())
print("Cleaned dataset saved as cleaned_employee_data.csv")




X = df[['Competition_Price', 'Item_Quantity', 'Sales_Amount']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("MAE:", mae)
print("R²:", r2)

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.legend()
plt.show()

X_single = df[['Sales_Amount']]
y_single = df['Price']

model_single = LinearRegression()
model_single.fit(X_single, y_single)
y_pred_single = model_single.predict(X_single)

plt.scatter(X_single, y_single, color='blue', label='Actual')
plt.plot(X_single, y_pred_single, color='red', label='Regression Line')
plt.xlabel('Sales_Amount')
plt.ylabel('Price')
plt.title('Linear Regression Line: Sales_Amount vs Price')
plt.legend()
plt.show()

#EDA analysis
# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sales_Amount', y='Price', data=df)
plt.title('Price vs Sales Amount')
plt.xlabel('Sales Amount')
plt.ylabel('Price')
plt.show()

# Multivariate Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Item_Quantity', y='Price', hue='Competition_Price', data=df)
plt.title('Relationship between Item Quantity and Price by Competition Price')
plt.xlabel('Item Quantity')
plt.ylabel('Price')
plt.legend(title='Competition Price')
plt.show()

#Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()







