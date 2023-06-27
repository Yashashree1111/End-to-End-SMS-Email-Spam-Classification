# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 18:49:30 2023

@author: yasha
"""

import pickle
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize

ps = PorterStemmer()


#preprocess = pickle.load(open("preprocess.plk","rb"))
vectorizer = pickle.load(open("vectorizer.plk","rb"))
model = pickle.load(open("model.plk","rb"))



st.title("SMS / Mail Spam Classifier")

def preprocessing(text):
    text = text.lower()
    text = word_tokenize(text)
    
    x = []
    # remove special characters
    for i in text:
        if i.isalnum():
            x.append(i)
            
    # remove stopwords from sentence
    for i in x:
        if i in stopwords.words("english"):
            x.remove(i)
            
    # stemming
    y = []
    for i in x:
        y.append(ps.stem(i))
        
        
    listToStr = ' '.join([str(elem) for elem in y])
    
    return listToStr


input_sms = st.text_area("Enter your SMS/Email")


    


if st.button(label="Predict"):
    
    
    
    try:
        
        # 1. Preprocess the input text 
         
        processed_input = preprocessing(input_sms)
         
        # 2. Vectorize the text
         
        vectorized_input =  vectorizer.transform([processed_input])
         
        # 3. predict output
         
        output = model.predict(vectorized_input)[0]
         
        if output == 1:
            
            st.header("Spam")
        else :
            st.header("Not Spam")
            
    except Exception as e:
        
        st.error("An error occurred: {}".format(str(e)))    
            
         
