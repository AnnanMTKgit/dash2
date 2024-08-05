import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
import base64
st.set_page_config(page_title='Dashboard',page_icon='📊',layout='wide')
st.subheader('Bienvenu sur le Tableau de Bord de Marlodj 🇸🇳')
st.markdown('###')

       
st.sidebar.image('logo.png',caption="",width=150)


# @st.cache_data
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()


# img = get_img_as_base64('background.jpg')

# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# #background-image: url('https://img.freepik.com/free-photo/dark-abstract-background_1048-1920.jpg?w=1800&t=st=1718946607~exp=1718947207~hmac=a4db4d36500b426a9916d8c86209d586bd1a763a69a3224a713594b9e16a8708');
# background-color: black;
# background-size: 100%;
# background-position: top left;
# background-repeat: no-repeat;
# background-attachment: local;
# }}

# [data-testid="stSidebar"] > div:first-child {{
# background: black;
# background-image: url("data:image/png;base64,{img}");
# background-position: center; 
# background-repeat: no-repeat;
# background-attachment: fixed;
# }}

# [data-testid="stHeader"] {{
# background: rgba(0,0,0,0);
# }}

# [data-testid="stToolbar"] {{
# right: 2rem;
# }}
# </style>
# """

# st.markdown(page_bg_img, unsafe_allow_html=True)

# Custom CSS for the sidebar background image


# Custom CSS
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
#     @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
#     @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;700&display=swap');

#     body {
#         font-family: 'Roboto', sans-serif;
#     }
#     h1, h2, h3, h4, h5, h6 {
#         font-family: 'Montserrat', sans-serif;
#     }
#     code, pre {
#         font-family: 'Source Code Pro', monospace;
#     }
# </style>
# """, unsafe_allow_html=True)



#st.title("")
st.sidebar.header("Configuration")


def option1(df_all,df_queue):
    st.subheader("GFA Global")
    
    





    pages = ['Congestion et localisation','Tableau Global', "Temps Moyen & Top10 d'Opération", 'Top5 Agences Plus Lentes',"Top5 Agences les plus Fréquentées"]
    option=st.radio(label=' ',options=pages,horizontal=True)


    
    # # Configure the figure to be static
    # config = {
    # 'staticPlot': True,  # Disable all interactive features
    # 'scrollZoom': False,  # Disable scroll zooming
    # 'displayModeBar': False  # Hide the mode bar
    # }

    

    # Afficher le contenu en fonction de la sélection
    if option == 'Congestion et localisation':
        
        Conjection(df_queue)

    elif option =='Tableau Global':
        
        HomeGlob(df_all,df_queue)

    elif option == "Temps Moyen & Top10 d'Opération":
        c1,c2=st.columns(2)
        
        plot_and_download(c1,GraphsGlob(df_all),'1')
        
        plot_and_download(c2,Top10_Type(df_queue),'2')


    elif option == 'Top5 Agences Plus Lentes':
        c1,c2=st.columns(2)
        plot_and_download(c1,Top5Agence(df_all,df_queue,"Temps Moyen d'Operation (MIN)",'Temps en minutes'),button_key='3')
        plot_and_download(c2,Top5Agence(df_all,df_queue,"Temps Moyen d'Attente (MIN)",'Temps en minutes'),button_key='4')
        
    elif option == "Top5 Agences les plus Fréquentées":
        c1,c2=st.columns(2)
        
        plot_and_download(c1,Top5Agence(df_all,df_queue,'Total Tickets','Nombre de Clients'),button_key='5')
        plot_and_download(c2,Top5Agence(df_all,df_queue,'Total Traités','Nombre de Clients'),button_key='6')
 
        
   


def option2(df_selected,df_queue):
    name_service=list(df_selected['NomService'].unique())
    st.subheader(f"{name_service}")
    
    

    pages = ['Top10 & Temps Moyen (Opération)',"Evolution Nbr Clients", "Performance Agents en Nbr de Clients", "Performance Agents en Temps"]
    
    option=st.radio(label=' ',options=pages,horizontal=True)
    
    
    if option == 'Top10 & Temps Moyen (Opération)':
        
        c1,c2=st.columns(2)
        plot_and_download(c1,Top10_Type_op(df_selected,df_all),button_key='7')
        plot_and_download(c2,TempsParType_op(df_selected,df_all),button_key='8')
        
        

    elif option == "Evolution Nbr Clients":
        
        c1,c2=st.columns([90,10])
        plot_and_download(c1,plot_line_chart(df_selected),button_key='9')
        
        
    
    elif option == "Performance Agents en Nbr de Clients":
        Graphs_pie(df_selected)

    elif option == "Performance Agents en Temps":
        Graphs_bar(df_selected)
        
        

start_date, end_date=date_range_selection()

current_date = datetime.now().date()
current_hour=datetime.now().hour
    # Vérifier si les dates sont égales à la date actuelle
if end_date == current_date and current_hour<18 and current_hour>7:
    # Forcer le rafraîchissement du cache en utilisant une clé spéciale
    
    df = get_sqlData(start_date, end_date)
else:
    # Charger les données en utilisant la fonction de mise en cache normale
    df = get_sqlData_cache(start_date, end_date)




#df_all,df_queue=get_sqlData(start_date, end_date)



if len(df)==0:
    st.warning('Données non disponible pour les Dates Choisies. Veuillez choisir des dates valides')
    st.warning("C'est peut être le Week-End ou un jour non ouvrable pour la banque")
    st.stop()
else:
    NomAgence=st.sidebar.multiselect(
        'Agences',
        options=df['NomAgence'].unique(),
        default=df['NomAgence'].unique()
    )
    df=df.query('NomAgence==@NomAgence')
    if df.empty:
        st.warning('Aucune Agence Sélectionnée')
        st.stop()
    df_all = df[df['UserName'].notna()].reset_index(drop=True)
    df_queue=df.copy()
    if df_queue.empty or df_all.empty:
        st.warning('Données Manquantes')
        st.stop()
    if 'selected_option' not in st.session_state:
        st.session_state['selected_option'] = "GFA Global"
    with st.sidebar:
        selected=option_menu(
            menu_title="Menu Principal",
            options=["GFA Global","Filtre"],
            icons=["globe","list-task"],
            menu_icon="cast",
            default_index=0
        )
    # Réinitialiser la page courante si l'option change
    if st.session_state['selected_option'] != selected:
        st.session_state['selected_option'] = selected
        st.session_state['option1_page'] = 1
        st.session_state['option2_page'] = 1
        st.session_state['option3_page'] = 1

    
    

    if selected=="GFA Global":
        
        #st.subheader(f"Page: {selected}")
        #Progressbar()
        option1(df_all,df_queue)
       

    if selected=="Filtre": # si la somme du Nbs de chaque service != Nbs Total alors UserId n'existe pas
        print('Waiting')    
        df_selected=filter1(df_all)
        if df_selected.empty:
            st.warning('Données Vides')
        else:
            option2(df_selected,df_queue)
            
            
