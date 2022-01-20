
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Depressiondataset=pd.read_csv("AiAssignment/dataset/Suicide_Detection.csv");


Depressiondataset.drop('Unnamed: 0',axis=1,inplace=True)
print(Depressiondataset.head())

Xfeatures=Depressiondataset['Clean_Text']
ylabels=Depressiondataset['class']


#Vectorizer

cv=CountVectorizer()
X=cv.fit_transform(Xfeatures)

#split dataset
X_train,X_test,y_train,y_test=train_test_split(X,ylabels,test_size=0.3,random_state=42)


#building model using Logistic Regression
nv_model= MultinomialNB()
nv_model.fit(X_train,y_train)

#Checking for accuracy
print("Accuracy: ",nv_model.score(X_test,y_test))



import pickle
# # Saving model to disk
pickle.dump(nv_model, open('Naivemodel.pkl','wb'))
naivemodel=pickle.load(open('Naivemodel.pkl','rb'))
