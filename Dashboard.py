import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import streamlit as st
import requests
import json

def request_prediction(model_uri, data):
    headers = {"Content-Type" : "application/json"}

    response = requests.request(
                method='POST', headers=headers, url=model_uri, data=data)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def display_categorical_feature(data):
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    option_order = data.value_counts().index
    ax.pie(data.value_counts(), labels = option_order, autopct='%.0f%%')
    st.pyplot(fig)


def display_numerical_feature(data):
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    sns.histplot(data = data, ax=ax)
    ax.tick_params(axis='x', labelrotation=45)
    st.pyplot(fig)
    
    df_option = pd.DataFrame(index=['>>'])
            
    df_option['Nbre clients']="{:.0f}".format(data.describe()['count'])
    df_option['Moyenne']="{:,.2f}".format(data.describe()['mean'])
    df_option['Ecart-type']="{:,.2f}".format(data.describe()['std'])
    df_option['Minimum']="{:,.2f}".format(data.describe()['min'])
    df_option['Médiane']="{:,.2f}".format(data.describe()['50%'])
    df_option['Maximum']="{:,.2f}".format(data.describe()['max'])
    
    st.write(df_option)


def display_feature_description(feature, df_column_desc):
    if "INSTAL_" in feature:
        df_feat_explanation = df_column_desc[df_column_desc['Table']=='installments_payments.csv']
        st.markdown("*INSTAL_ = For installments payments*")
        
    elif "CC_" in feature:
        df_feat_explanation = df_column_desc[df_column_desc['Table']=='credit_card_balance.csv']
        st.markdown("*CC_ = For credit card balance*")
        
    elif "BURO_" in feature or "ACTIVE_" in feature or "CLOSED_" in feature:
        df_feat_explanation = df_column_desc[(df_column_desc['Table']=='bureau.csv') | (df_column_desc['Table']=='bureau_balance.csv')]
        if "ACTIVE_"in feature:
            st.markdown("*ACTIVE_ = For active credits with other financial institutions*")
        elif "CLOSED_" in feature:
            st.markdown("*CLOSED_ = For closed credits with other financial institutions*")
        elif "BURO_" in feature:
            st.markdown("*BURO_ = For credits with other financial institutions*")   
                                
    elif "PREV_" in feature or "APPROVED_" in feature or "REFUSED_" in feature:
        df_feat_explanation = df_column_desc[df_column_desc['Table']=='previous_application.csv']
        if "PREV_"in feature:
            st.markdown("*PREV_ = For previous applications with 'Prêt à dépenser'*")
        elif "APPROVED_" in feature:
            st.markdown("*APPROVED_ = For previous approved applications with 'Prêt à dépenser'*")
        elif "REFUSED_" in feature:
            st.markdown("*REFUSED_ = For previous refused applications with 'Prêt à dépenser'*") 
                
    elif "POS_" in feature:
        df_feat_explanation = df_column_desc[df_column_desc['Table']=='POS_CASH_balance.csv']
        st.markdown("*POS_ = For POS Cash balance*")
        
    else:
        df_feat_explanation = df_column_desc[df_column_desc['Table']=='application_{train|test}.csv']
    
    
    if "MEAN" in feature:
        st.markdown("*MEAN = Mean of*")
    if "SUM" in feature:
        st.markdown("*SUM = Sum of*")
    if "VAR" in feature:
        st.markdown("*VAR = Variance of*")
    if "MIN" in feature:
        st.markdown("*MIN = Minimum of*")
    if "MAX" in feature:
        st.markdown("*MAX = Maximum of*")
    if "SIZE" in feature:
        st.markdown("*SIZE = Count of*")
        
    for var in df_feat_explanation['Row']:
        if var in feature:
            st.markdown("*{} = {}*".format(var, df_feat_explanation[df_feat_explanation['Row']==var].iloc[0]['Description']))
    


def main():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
        
    flask_uri_prediction = 'https://ben10.pythonanywhere.com/prediction'
    flask_uri_shap_values = 'https://ben10.pythonanywhere.com/shap_values'
    
    df_info = pd.read_csv('df_application_test_sample.csv', sep=',')
    df_info = df_info.drop('Unnamed: 0', axis=1)
    
    df_model = pd.read_csv('df_final_app_test_sample.csv', sep=',')
    df_model = df_model.set_index('SK_ID_CURR')
    

    with st.sidebar:
        st.header("Tableau de bord de suivi client")
        
        st.write('')
        user_input = st.selectbox(
                        "Indiquer l'ID du client :",
                        df_info['SK_ID_CURR'])
        
        st.write("Vous avez sélectionné l'ID suivant : ", str(user_input))
        
        st.write('')
        st.write('')
        
        page_to_display = st.radio("Quelle page souhaitez vous afficher ?",
                                    ("Informations descriptives du client", "Prédiction sur sa capacité de remboursement",
                                    "Comparaison par rapport à d'autres clients"))
    
    df_user = df_info[df_info['SK_ID_CURR'] == int(user_input)]
    df_info_transpose = df_info[df_info['SK_ID_CURR'] == int(user_input)].transpose()
    df_info_transpose = df_info_transpose.astype('str')
    df_info_transpose.rename(columns={ df_info_transpose.columns[0] : "Information du client" }, inplace=True)

    default_options = ['CODE_GENDER', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                    'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
    
    
    if page_to_display=="Informations descriptives du client":
        st.header("Informations descriptives du client :")
        options = st.multiselect("Choisissez les informations à afficher :", df_info_transpose.index, default_options)
        st.write(df_info_transpose.loc[options,:])
        
        st.write('')
        st.write('')    
        with st.expander("Cliquez ici pour voir la totalité des informations descriptives disponibles sur le client :"):
            st.write(df_info_transpose)
    
    elif page_to_display=="Prédiction sur sa capacité de remboursement":        
        st.header("Prédiction sur sa capacité de remboursement :")
        
        with st.spinner('La prédiction est en cours de calcul...'):
            data = df_model.loc[[int(user_input)],:]
            result = data.to_json(orient="split")
            
            dict_prediction = request_prediction(flask_uri_prediction, result)        
            dict_shap_values = request_prediction(flask_uri_shap_values, result)
        
        st.markdown(
            "Le client avec l'ID {} a ***{:.0%} de chance*** de rembourser son prêt.".format(str(user_input), dict_prediction["predict_proba"][0][0]))
        
        df_feature_importance_class_0 = pd.DataFrame(data=dict_shap_values["data"], index=dict_shap_values["index"], columns=dict_shap_values["columns"])
        
        df_user_feat_imp = df_feature_importance_class_0.loc[int(user_input), :]
        
        top3_class_1 = df_user_feat_imp.sort_values(ascending=False).head(3)
        top3_class_0 = df_user_feat_imp.sort_values(ascending=True).head(3)
        
        df_column_desc = pd.read_excel('Descriptions colonnes.xlsx')
        
        st.write('')
        st.subheader("Les 3 principales raisons d'un remboursement du prêt :")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = Figure(figsize=(4, 4))
            ax = fig.subplots()
            ax.pie(top3_class_1, labels = top3_class_1.index, autopct='%.0f%%')
            st.pyplot(fig)
               
        
        st.markdown("*Données du client sur ces variables :*")
        i = 1
        for index in top3_class_1.index:
            st.write(str(i) + ' - ' + str(index) + ' : ' + str(np.round(df_model.loc[int(user_input), index], 2)))
            i = i + 1
           
        with st.expander("Cliquez ici pour voir la signification des 3 variables ci-dessus :"):
            i = 1
            for index in top3_class_1.index:
                st.markdown("**{} - {} :**".format(str(i), index))
                i = i + 1
                display_feature_description(index, df_column_desc)                
                st.write('')                 
                
          
        st.write('')
        st.write('')
        st.subheader("Les 3 principales raisons d'un non remboursement du prêt :")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = Figure(figsize=(4, 4))
            ax = fig.subplots()
            ax.pie(abs(top3_class_0), labels = top3_class_0.index, autopct='%.0f%%')
            st.pyplot(fig)
        
        st.markdown("*Données du client sur ces variables :*")
        i = 1
        for index in top3_class_0.index:
            st.write(str(i) + ' - ' + str(index) + ' : ' + str(np.round(df_model.loc[int(user_input), index], 2)))
            i = i +1
        
        with st.expander("Cliquez ici pour voir la signification des 3 variables ci-dessus :"):
            i = 1
            for index in top3_class_0.index:
                st.markdown("**{} - {} :**".format(str(i), index))
                i = i + 1
                display_feature_description(index, df_column_desc)                
                st.write('')  
        
    
    elif page_to_display=="Comparaison par rapport à d'autres clients":
        st.header("Comparaison par rapport à d'autres clients :")
    
        comparaison_type = st.radio("Quelle comparaison souhaitez vous faire ?",
                                    ("Par rapport à l'ensemble des clients", "Par rapport à un groupe de clients similaires"))

        if comparaison_type == "Par rapport à l'ensemble des clients" :
            
            st.write('')
            options = st.multiselect("Choisissez les informations à comparer :", df_info_transpose.index, default_options)
            st.write('')

            k=0
            
            while k < len(options):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Comparaison concernant {} :**".format(options[k]))
                    st.markdown("*Donnée du client : {}*".format(str(df_info_transpose.loc[options[k], df_info_transpose.columns[0]])))
                    if df_info[options[k]].dtype == 'object':
                        display_categorical_feature(df_info[options[k]])
                        
                    else:
                        display_numerical_feature(df_info[options[k]])
                    
                    st.write('')
                    st.write('')
                    
                        
                if k+1 <len(options):
                    with col2:
                        st.markdown("**Comparaison concernant {} :**".format(options[k+1]))
                        st.markdown("*Donnée du client : {}*".format(str(df_info_transpose.loc[options[k+1], df_info_transpose.columns[0]])))
                        if df_info[options[k+1]].dtype == 'object':
                            display_categorical_feature(df_info[options[k+1]])
                            
                        else:
                            display_numerical_feature(df_info[options[k+1]])
                        
                        st.write('')
                        st.write('')
                k = k+2
                
                
        else:
            st.write('')
            options = st.multiselect("Clients similaires par rapport à :", df_info_transpose.index, ['CODE_GENDER', 'AMT_INCOME_TOTAL'])
            
            filter = []
            
            for option in options:
                if df_info[option].dtype == 'object':
                    st.markdown("- *Donnée du client concernant {} : {}*".format(option, df_info_transpose.loc[option, df_info_transpose.columns[0]]))
                    filter.append(df_user.loc[df_user.index[0],option])
                
                else:
                    st.markdown("- *Donnée du client concernant {} : {:,}*".format(option, float(df_info_transpose.loc[option, df_info_transpose.columns[0]])))
                    
                    if df_info[option].dtype == 'int64':
                        min_value = int(np.int64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*0.7)
                        max_value = int(np.int64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*1.3) 
                        
                        lower_bound = int(np.int64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*0.9)
                        upper_bound = int(np.int64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*1.1) 
                    else:
                        min_value = float(np.float64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*0.7)
                        max_value = float(np.float64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*1.3)
                        
                        lower_bound = float(np.float64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*0.9)
                        upper_bound = float(np.float64(df_info_transpose.loc[option, df_info_transpose.columns[0]])*1.1)
                    
                    
                    values = st.slider("Vous pouvez ajuster la plage de similarité à appliquer pour cet attribut : ", min_value, max_value,
                                      (lower_bound, upper_bound))
                    
                    st.write("Pour l'attribut {}, vous avez choisi d'appliquer la plage de similarité suivante : [{:,.2f}, {:,.2f}]".format(option, values[0], values[1]))
                    
                    filter.append(values)
                    
        
            
            df_info_filtered = df_info
            
            for num, option in enumerate(options):
                if df_info[option].dtype == 'object':
                    df_info_filtered = df_info_filtered[df_info_filtered[option] == filter[num]]
                else:
                    df_info_filtered = df_info_filtered[(df_info_filtered[option] >= filter[num][0]) & (df_info_filtered[option] <= filter[num][1])]
            
            st.write('')
            st.write('')
            st.write('')
            options = st.multiselect("Choisissez maintenant les informations que vous souhaitez comparer par rapport au groupe de clients\
                                     similaires sélectionné :", df_info_transpose.index, default_options)
            st.write('')
            
            display_btn = st.button("Cliquez ici pour afficher la comparaison souhaitée")
            st.write('')
            st.write('')
            
            if display_btn:
                k=0
                
                while k < len(options):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Comparaison concernant {} :**".format(options[k]))
                        st.markdown("*Donnée du client : {}*".format(str(df_info_transpose.loc[options[k], df_info_transpose.columns[0]])))
                        if df_info[options[k]].dtype == 'object':
                            display_categorical_feature(df_info_filtered[options[k]])
                            
                        else:
                            display_numerical_feature(df_info_filtered[options[k]])
                        
                        st.write('')
                        st.write('')
                            
                    if k+1 <len(options):
                        with col2:
                            st.markdown("**Comparaison concernant {} :**".format(options[k+1]))
                            st.markdown("*Donnée du client : {}*".format(str(df_info_transpose.loc[options[k+1], df_info_transpose.columns[0]])))
                            if df_info[options[k+1]].dtype == 'object':
                                display_categorical_feature(df_info_filtered[options[k+1]])
                                
                            else:
                                display_numerical_feature(df_info_filtered[options[k+1]])
                            
                            st.write('')
                            st.write('')
                            
                    k = k+2
        
                
if __name__ == '__main__':
    main()