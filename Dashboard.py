import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import streamlit as st
import requests
import json

def request_prediction(model_uri, data_json):
    headers = {"Content-Type" : "application/json"}

    # data_json = {'data': data}
    # response = requests.request(
    #    method='POST', headers=headers, url=model_uri, json=data_json)

    response = requests.request(
                method='POST', headers=headers, url=model_uri, data=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()



def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    
    df_info = pd.read_csv('df_app_test.csv', sep=',')
    df_info = df_info.drop('Unnamed: 0', axis=1)

    user_input = st.text_input(
            "Indiquer l'ID du client :"
        )
    
    df_model = pd.read_csv('df_final_test.csv', sep=',')
    df_model = df_model.set_index('SK_ID_CURR')
    
    df_feature_importance_class_0 = pd.read_csv('df_feature_importance_class_0.csv', sep=',')
    df_feature_importance_class_0 = df_feature_importance_class_0.set_index('SK_ID_CURR')
    
    predict_btn = st.button('Prédire')
    if predict_btn:
        
        data = df_model.loc[[int(user_input)],:]
        result = data.to_json(orient="split")
        
        pred = request_prediction(MLFLOW_URI, result)
        
        
        st.write(
            'Ce client a {:.0%} de chance de rembourser son prêt.'.format(pred[0][0]))
        
        df_user_feat_imp = df_feature_importance_class_0.loc[int(user_input), :]
        
        top3_class_1 = df_user_feat_imp.sort_values(ascending=False).head(3)
        top3_class_0 = df_user_feat_imp.sort_values(ascending=True).head(3)
       
       
        st.write("Les 3 principales raisons d'un remboursement du prêt :")
        
        fig = Figure(figsize=(4, 4))
        ax = fig.subplots()
        ax.pie(top3_class_1, labels = top3_class_1.index, autopct='%.0f%%')
        st.pyplot(fig)
        
        for index in top3_class_1.index:
            st.write(str(index) + ' : ' + str(df_model.loc[int(user_input), index]))
    
        st.write("Les 3 principales raisons d'un non remboursement du prêt :")
        
        fig = Figure(figsize=(4, 4))
        ax = fig.subplots()
        ax.pie(abs(top3_class_0), labels = top3_class_0.index, autopct='%.0f%%')
        st.pyplot(fig)
        
        for index in top3_class_0.index:
            st.write(str(index) + ' : ' + str(df_model.loc[int(user_input), index]))
        
    

    df_user = df_info[df_info['SK_ID_CURR'] == int(user_input)]
    df_info_transpose = df_info[df_info['SK_ID_CURR'] == int(user_input)].transpose()
    df_info_transpose = df_info_transpose.astype('str')
    df_info_transpose.rename(columns={ df_info_transpose.columns[0] : "Information du client" }, inplace=True)

    default_options = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                    'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

    options = st.multiselect("Filtre sur les informations disponibles :", df_info_transpose.index, default_options)
    df_info_transpose.loc[options,:]

    "Ensemble des informations sur le client :"
    st.write(df_info_transpose)

    comparaison_type = st.radio("Quelle comparaison souhaitez vous faire ?",
                                ("Par rapport à l'ensemble des clients", "Par rapport à un groupe de clients similaires"))

    if comparaison_type == "Par rapport à l'ensemble des clients" :
        
        options = st.multiselect("Informations à comparer :", df_info_transpose.index, default_options)

        k=0
        
        while k < len(options):
            col1, col2 = st.columns(2)
            with col1:
                "Comparaison concernant : " + options[k]
                "Donnée du client : " + str(df_info_transpose.loc[options[k], df_info_transpose.columns[0]])
                if df_info[options[k]].dtype == 'object':
                    fig = Figure(figsize=(4, 4))
                    ax = fig.subplots()
                    option_order = df_info[options[k]].value_counts().index
                    ax.pie(df_info[options[k]].value_counts(), labels = option_order, autopct='%.0f%%')
                    st.pyplot(fig)
                    
                else:
                    fig = Figure(figsize=(4, 4))
                    ax = fig.subplots()
                    sns.histplot(data = df_info[options[k]], ax=ax)
                    st.pyplot(fig)
                    
                    st.write(df_info[options[k]].describe())
                    
            if k+1 <len(options):
                with col2:
                    "Comparaison concernant : " + options[k+1]
                    "Donnée du client : " + str(df_info_transpose.loc[options[k+1], df_info_transpose.columns[0]])
                    if df_info[options[k+1]].dtype == 'object':
                        fig = Figure(figsize=(4, 4))
                        ax = fig.subplots()
                        option_order = df_info[options[k+1]].value_counts().index
                        ax.pie(df_info[options[k+1]].value_counts(), labels = option_order, autopct='%.0f%%')
                        st.pyplot(fig)
                        
                    else:
                        fig = Figure(figsize=(4, 4))
                        ax = fig.subplots()
                        sns.histplot(data = df_info[options[k+1]], ax=ax)
                        st.pyplot(fig)
                        
                        st.write(df_info[options[k+1]].describe())
                    
            k = k+2
            
            
    else:
        options = st.multiselect("Clients similaires par rapport à :", df_info_transpose.index, ['CODE_GENDER', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL'])
        
        filter = []
        
        for option in options:
            if df_info[option].dtype == 'object':
                "Donnée du client concernant " + option + ' : ' + df_info_transpose.loc[option, df_info_transpose.columns[0]]
                filter.append(df_user.loc[df_user.index[0],option])
            
            else:
                "Donnée du client concernant " + option + " : " + df_info_transpose.loc[option, df_info_transpose.columns[0]]
                
                if df_info[option].dtype == 'int64':
                    lower_bound = int(np.int64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*0.9)
                    upper_bound = int(np.int64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*1.1) 
                else:
                    lower_bound = float(np.float64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*0.9)
                    upper_bound = float(np.float64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*1.1)
            
                values = st.slider("Choisissez le segment de similarité à appliquer pour cet attribut : ", df_info[option].min(), df_info[option].max(),
                                (lower_bound, upper_bound))
                
                filter.append(values)
                
        filter
        
        df_info_filtered = df_info
        
        for num, option in enumerate(options):
            if df_info[option].dtype == 'object':
                df_info_filtered = df_info_filtered[df_info_filtered[option] == filter[num]]
            else:
                df_info_filtered = df_info_filtered[(df_info_filtered[option] >= filter[num][0]) & (df_info_filtered[option] <= filter[num][1])]
        
        df_info_filtered
        
        options = st.multiselect("Informations à comparer :", df_info_transpose.index, default_options)
        
        k=0
        
        while k < len(options):
            col1, col2 = st.columns(2)
            with col1:
                "Comparaison concernant : " + options[k]
                "Donnée du client : " + str(df_info_transpose.loc[options[k], df_info_transpose.columns[0]])
                if df_info[options[k]].dtype == 'object':
                    fig = Figure(figsize=(4, 4))
                    ax = fig.subplots()
                    option_order = df_info_filtered[options[k]].value_counts().index
                    ax.pie(df_info_filtered[options[k]].value_counts(), labels = option_order, autopct='%.0f%%')
                    st.pyplot(fig)
                    
                else:
                    fig = Figure(figsize=(4, 4))
                    ax = fig.subplots()
                    sns.histplot(data = df_info_filtered[options[k]], ax=ax)
                    st.pyplot(fig)
                    
                    st.write(df_info_filtered[options[k]].describe())
                    
            if k+1 <len(options):
                with col2:
                    "Comparaison concernant : " + options[k+1]
                    "Donnée du client : " + str(df_info_transpose.loc[options[k+1], df_info_transpose.columns[0]])
                    if df_info[options[k+1]].dtype == 'object':
                        fig = Figure(figsize=(4, 4))
                        ax = fig.subplots()
                        option_order = df_info_filtered[options[k+1]].value_counts().index
                        ax.pie(df_info_filtered[options[k+1]].value_counts(), labels = option_order, autopct='%.0f%%')
                        st.pyplot(fig)
                        
                    else:
                        fig = Figure(figsize=(4, 4))
                        ax = fig.subplots()
                        sns.histplot(data = df_info_filtered[options[k+1]], ax=ax)
                        st.pyplot(fig)
                        
                        st.write(df_info_filtered[options[k+1]].describe())
                    
            k = k+2
        
                
if __name__ == '__main__':
    main()