#!/usr/bin/env python
# coding: utf-8

# # *Problem Statement*

# Create a machine learning model that can predict the genre of a
# movie based on its plot summary or other textual information. You
# can use techniques like TF-IDF or word embeddings with classifiers
# such as Naive Bayes, Logistic Regression, or Support Vector
# Machines.

# # Step #1: Importing Data

# In[31]:


with open(r"C:\Users\pruth\Downloads\Codsoft\Task1\Genre Classification Dataset\train_data.txt",'r',errors='ignore') as file:
    text_data=file.read()


# In[32]:


text_data


# In[33]:


# Split text data into rows
lines = text_data.split('\n')

# Split each row into individual values
data = [line.split(':::') for line in lines]


# In[34]:


data


# #### Converting the text file to CSV

# In[35]:


import csv

# Specify the path for the output CSV file
output_file = "output.csv"

# Open the CSV file in write mode and write the data
with open(output_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header row if your data has one
    # For example, if the first row of your text file is the header:
    # csv_writer.writerow(data[0])
    
    # Write the remaining rows
    csv_writer.writerows(data)


# # Step #2: Data Cleaning

# In[36]:


import pandas as pd
df=pd.read_csv('outputp.csv',encoding='ISO-8859-1')


# In[37]:


df


# In[38]:


X=df['Description']


# In[39]:


X


# In[40]:


y=df['Genre']


# In[41]:


y


# # Step #3: Vectorization

# In[42]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[43]:


tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,  # Limit the number of features (words) to 5000
    stop_words='english'  # Remove common English stop words
)


# In[44]:


X_tfidf = tfidf_vectorizer.fit_transform(X)


# In[45]:


X_tfidf


# # Step #4: Model Training

# In[46]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


# In[47]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[48]:


# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')


# In[49]:


# Transform the textual data into TF-IDF vectors
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[50]:


# Define a Logistic Regression classifier
# Manually choose hyperparameters here
clf = LogisticRegression(max_iter=1000, C=1.0, penalty='l2')


# In[51]:


# Fit the classifier on the training data
clf.fit(X_train_tfidf, y_train)


# # Step #5: Model Prediction

# In[52]:


# Make predictions on the test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1-score for all classes
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

report = classification_report(y_test, y_pred, zero_division=0)

print(f"Accuracy: {accuracy}")
print(report)

# Print precision, recall, and F1-score for each class
for i, genre in enumerate(clf.classes_):
    print(f"Genre: {genre}")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1-Score: {f1[i]}")
    print(f"Support: {support[i]}")
    print()


# In[ ]:




