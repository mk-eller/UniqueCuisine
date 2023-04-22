import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle

fileName1 = "ModelData.csv"
data = pd.read_csv(fileName1)

data.drop(['Unnamed: 0'], axis=1, inplace=True)

data = data.dropna()

x = data.ingredients
y = data.tags
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
rfc = Pipeline([('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', RandomForestClassifier())
 ])
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

pickle.dump(rfc, open('model.pkl','wb'))






