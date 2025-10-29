import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import joblib

#load dataset
df = pd.read_csv("data/products.csv")

#converting column Product Title to string,converting text to lower cases and striping empty spaces in the beggining and end
df['Product Title'] = df['Product Title'].astype(str).str.lower().str.strip()
#converting column Category Label to string and converting text to lowr cases
df[' Category Label'] = df[' Category Label'].astype(str).str.lower()


df = df.dropna()#removing missing valuers from all columns


#removing columns that are not useful for our model
df = df.drop(columns=['product ID', 'Merchant ID', '_Product Code', 'Number_of_Views', 'Merchant Rating', ' Listing Date  '])

#spliting data on feature and label
x = df[["Product Title"]]
y = df[" Category Label"]


#converting string data into numerical data using tfidfVectorizer 
preprocesor = ColumnTransformer([
        ("title",TfidfVectorizer(),"Product Title"),
])

pipeline= Pipeline([
       ("preprocesing", preprocesor),
       ("Classifier",LogisticRegression())])

pipeline.fit(x,y)

#saving trained model

joblib.dump(pipeline,"data/model/product classifier model.pkl")

