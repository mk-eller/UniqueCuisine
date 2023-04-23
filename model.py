import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.tree import DecisionTreeClassifier

fileName1 = "ModelData.csv"
data = pd.read_csv(fileName1)

data.drop(['Unnamed: 0'], axis=1, inplace=True)

data = data.dropna()

x = data.ingredients
y = data.tags
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)

#This model is pulled directly from my Jupyter Notebook: Model Building and Evaluation Final. It is
#the one I decided was my best.

dtc = Pipeline([('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', DecisionTreeClassifier())
 ])
dtc.set_params(clf__random_state=42,
              clf__max_features=600)
dtc.fit(x_train, y_train)
y_pred4 = dtc.predict(x_test)

#Pickling my model to use in my prediction function on Unique.py

pickle.dump(dtc, open('model.pkl','wb'))






