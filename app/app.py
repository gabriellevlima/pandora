import tensorflow as tf
from tensorflow import keras as ke
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import re
import string
import nltk
import sklearn
import plotly.express as px
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import h5py 
import gensim
from nltk.stem.porter import *
import joblib,os
import base64
from io import BytesIO


st.sidebar.title("Menu")

def convert_input(user_input):
    # clean text
    ps = PorterStemmer() 
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', user_input) 
    review = review.lower() 
    review = review.split() 
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] 
    review = ' '.join(review) 
    corpus.append(review) 

    #vectorizer
    vectorizer = pickle.load(open("vector.pickel", "rb"))
    test_data_features = vectorizer.transform(corpus)
    return test_data_features

def loading_prediction_models(model_file):
    loading_prediction_models = joblib.load(open(os.path.join(model_file),'rb'))
    return loading_prediction_models

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'


def run():

    #st.sidebar.info(' ')    
    st.set_option('deprecation.showfileUploaderEncoding', False) 
    premium = st.sidebar.radio('Tipo de Assinatura:',('Versão Gratuita','Versão Premium'),index=1,key=2)
    
    st.title("Detector de Fake News")
    st.subheader('Hackathon Visagio')

    if premium == 'Versão Gratuita':
        add_selectbox = st.sidebar.selectbox("Como você gostaria de prever?", ("Online", "Arquivo Txt"))    
        all_ml_models = ['Decision Tree','Naive Bayes'] 
        model_choice = st.sidebar.selectbox("Com qual modelo?", all_ml_models) 

        if add_selectbox == "Online":
            user_input = st.text_area('Insira o texto',height = 300)
            output = ""
   
            if st.button('Run'):
                text = convert_input(user_input)
                if model_choice == "Naive Bayes":
                    predictor = loading_prediction_models('NaiveB.sav')
                    prediction = predictor.predict(text)
                    if prediction == 1:
                        st.success('Não é fake news')
                    elif prediction == 0:
                        st.error('É fake news')
                elif model_choice == "Decision Tree":
                    predictor = loading_prediction_models('DecisionT.sav')
                    prediction = predictor.predict(text)
                    if prediction == 1:
                        st.success('Não é fake news')
                    elif prediction == 0:
                        st.error('É fake news')


            
        elif add_selectbox == "Arquivo Txt":        
            output = ""
            file_buffer = st.file_uploader("Carregue o texto", type=["txt"]) 
        
            if st.button('Run'):
                user_input = file_buffer.read() 
                st_version = st.__version__  
                versions = st_version.split('.')           
                if int(versions[1]) > 67:
                    user_input = user_input.decode('utf-8')
            
                    text = convert_input(user_input)
                    if model_choice == "Naive Bayes":
                        predictor = loading_prediction_models('NaiveB.sav')
                        prediction = predictor.predict(text)
                        if prediction == 1:
                            st.success('Não é fake news')
                        elif prediction == 0:
                            st.error('É fake news')
                    elif model_choice == "Decision Tree":
                        predictor = loading_prediction_models('DecisionT.sav')
                        prediction = predictor.predict(text)
                        if prediction == 1:
                            st.success('Não é fake news')
                        elif prediction == 0:
                            st.error('É fake news')
     
    elif premium == 'Versão Premium':
        input = st.text_input('Sobre o que/quem você gostaria de pesquisar?')
        if st.button('Run'):
            data = pd.read_csv('Dados.csv')
            random = data[data['Texto'].str.contains(input)].sample(30)

            st.subheader("Dados")
            st.markdown(get_table_download_link(random), unsafe_allow_html=True)

            st.subheader('Top 5 notícias acessadas:')

            my_expander = st.beta_expander("Expand", expanded=True)
            with my_expander:
                fake = random[random['Verificador'] == 'Falso']
                top = fake.sort_values(by=['Popularidade'],ascending=False ).head(5)
                rank = top[['Texto', 'Verificador']]
                st.table(rank)

            st.subheader('Análise')
            value_counts = random['Verificador'].value_counts()
            fig = px.pie(names=value_counts.index, values=value_counts.values,  title='Proporção das notícias:')
            st.plotly_chart(fig)
            counts = fake['Mês'].value_counts()
            figu = px.bar(fake, x=counts.index, y=counts.values, title="Distribuição de notícias falsas",  labels=dict(x="Mês", y="Quantidade de notícias falsas"))
            st.plotly_chart(figu)


    st.beta_container()


if __name__ == "__main__":
    #modelo()
    run()
              
    
