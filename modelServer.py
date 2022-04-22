# from crypt import methods
from operator import methodcaller
from flask import Flask, request
# import numpy as np 
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import  stopwords
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def pred_model(inp):
     
    prediction = LG.predict(inp)
    return prediction

# @app.route('/predict', methods=['POST'])
# def get_text():
    
#     return 0
    

@app.route('/')
def index():
    return '<p>Machine Learning Inference</p>'

@app.route('/predict', methods=["GET"])
def get_text():
    arg = request.args
    msgdict=arg.to_dict()
    msg=[msgdict["address"]]
    inp=tf_vec.transform(msg)
    outp=pred_model(inp)
    print (msgdict["address"],'\n\n',outp)
    print('\n',type(outp))
    return str(outp[0])

if __name__ == "__main__":

    train = pd.read_csv("train.csv")

    train['length'] = train ['comment_text'].str.len()

    train['comment_text'] = train['comment_text'].str.lower()

    # Replace email addresses with 'email'
    train['comment_text'] = train['comment_text'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')

    # Replace URLs with 'webaddress'
    train['comment_text'] = train['comment_text'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

    # Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
    train['comment_text'] = train['comment_text'].str.replace(r'£|\$', 'dollers')
        
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    train['comment_text'] = train['comment_text'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')
        
    # Replace numbers with 'numbr'
    train['comment_text'] = train['comment_text'].str.replace(r'\d+(\.\d+)?', 'numbr')

    train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join(term for term in x.split() if term not in string.punctuation))

    stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])
    train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

    lem=WordNetLemmatizer()
    train['comment_text'] = train['comment_text'].apply(lambda x: ' '.join(lem.lemmatize(t) for t in x.split()))

    train['clean_length'] = train.comment_text.str.len()
    cols_target = ['malignant','highly_malignant','rude','threat','abuse','loathe']
    target_data = train[cols_target]
    train['bad'] =train[cols_target].sum(axis =1)

    train['bad'] = train['bad'] > 0 
    train['bad'] = train['bad'].astype(int)
    tf_vec = TfidfVectorizer(max_features = 10000, stop_words='english')
   
    features = tf_vec.fit_transform(train['comment_text'])
    x = features
    y=train['bad']
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=56,test_size=.30)
   

    LG = LogisticRegression(C=1, max_iter = 3000)

    LG.fit(x_train, y_train)

    app.run(debug=True)


