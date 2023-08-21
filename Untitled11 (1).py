#!/usr/bin/env python
# coding: utf-8

# # import library for read the dataset

# In[1]:


import pandas as pd


# In[23]:


df1 = pd.read_csv('Admission_Predict.csv')

df1


# In[25]:


df1.head(50)


# In[26]:


df1.tail(50)


# In[28]:


df2 = pd.read_csv("Admission_Predict_Ver1.1.csv")

df2


# In[29]:


df2.head(50)


# In[40]:


df2.tail(50)


# In[99]:


df3 = pd.concat([df1, df2.iloc[400:499]], ignore_index=True)

df3


# In[38]:


df3.tail(101)


# # Analyze the data

# In[41]:


df3.isnull()


# In[42]:


df3.isnull().sum()


# In[52]:


print(df3.columns)


# In[45]:


df3.describe()


# In[46]:


print(df3.dtypes)


# In[47]:


df3.count()


# In[48]:


df3.nunique


# In[50]:


# Assuming df3 is your DataFrame
top_50_students = df3.nlargest(50, 'GRE Score')

# Now, top_50_students contains the top 50 students based on GRE Score
top_50_students


# In[51]:


University = df3[df3['University Rating'] == 5]

University


# In[62]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a countplot
plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.countplot(x="Chance of Admit ", data=df3, palette="Set2")

# Customize the plot
plt.title('Count of Graduate Admissions Prediction')
plt.xlabel('Chance of Admit')
plt.ylabel('Count')

# Show the plot
plt.show()

# Find and print the data associated with a specific category (e.g., Chance of Admit = 0.8)
category_to_find = 0.8
data_for_category = df3[df3['Chance of Admit '] == category_to_find]

# Print the data for the selected category
print("Data for Chance of Admit =", category_to_find)


# In[63]:


print(data_for_category)


# # Machine Learning model

# In[65]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error


# Select the relevant columns
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
target = 'Chance of Admit '

# Split the data into features (X) and target (y)
X = df3[features]
y = df3[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but can be helpful for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")

# Visualize feature importances
feature_importances = model.feature_importances_
plt.bar(features, feature_importances)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


# In[91]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Select the relevant columns
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
target = 'Chance of Admit '

# Create a binary classification target variable based on 'Chance of Admit'
df3['Admitted'] = (df3['Chance of Admit '] >= 0.7).astype(int)

# Split the data into features (X) and the binary target (y)
X = df3[features]
y = df3['Admitted']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but can be helpful for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert X_test back to a DataFrame
X_test = pd.DataFrame(X_test, columns=features)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)



# In[92]:


y_pred


# In[93]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Show the data, actual admissions, and predictions
data_to_show = X_test.copy()
data_to_show['Actual Admitted'] = y_test
data_to_show['Predicted Admitted'] = y_pred

print("Data with Actual and Predicted Admissions:\n", data_to_show)


# In[86]:


print(f"\nAccuracy: {accuracy}")


# In[89]:


print("Confusion Matrix:\n", confusion)


# In[90]:


print("Classification Report:\n", classification_rep)


# In[94]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Calculate ROC curve
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate AUC score
auc = roc_auc_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[95]:


# Get the coefficients (importances) of the features
coefficients = model.coef_[0]

# Create a DataFrame to store the coefficients along with feature names
coefficients_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})

# Sort the coefficients by magnitude (absolute value)
coefficients_df['Absolute Coefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='Absolute Coefficient', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(coefficients_df['Feature'], coefficients_df['Absolute Coefficient'], color='skyblue')
plt.xlabel('Absolute Coefficient')
plt.ylabel('Feature')
plt.title('Feature Importances for Chance of Admit Prediction')
plt.gca().invert_yaxis()  # Invert the y-axis to display the most important feature at the top
plt.show()

# Display the sorted coefficients DataFrame
print("Feature Importances:\n", coefficients_df)


# In[ ]:




