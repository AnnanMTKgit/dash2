import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
import base64
st.set_page_config(page_title='Dashboard',page_icon='📊',layout='wide')
st.sidebar.subheader('Bienvenu sur le Tableau de Bord de Marlodj 🇸🇳')
       
st.sidebar.image('logo.png',caption="",width=150)


st.markdown("""
    <style>
    @media (prefers-color-scheme: dark) {
        /* Dark mode styles */
        body {
            color: #E0E0E0;  /* Light gray text for dark theme */
            background-color: #000000;
        }
    }

    @media (prefers-color-scheme: light) {
        /* Light mode styles */
        body {
            color: #000000;  /* Dark text for light theme */
            background-color: #FFFFFF;
        }
    }
    </style>
""", unsafe_allow_html=True)

#st.title("")
st.sidebar.header("Configuration")


def option1(df_all,df_queue):
    
    pages = ['Congestion et localisation','Tableau Global','Agence & Queue' ,"Temps Moyen & Type d'Opération", 'Agences Plus Lentes',"Agences les plus Fréquentées"]
    option=st.radio(label=' ',options=pages,horizontal=True)

    # Afficher le contenu en fonction de la sélection
    if option == 'Congestion et localisation':
        
        Conjection(df_queue)

    elif option =='Tableau Global':
        
        HomeGlob(df_all,df_queue)
    elif option=='Agence & Queue':
        chart1=stacked_chart(df_all,'TempsAttenteReel','NomAgence',"Catégorisation du Temps d'Attente")
        st.altair_chart(chart1)
        chart2=stacked_chart(df_all,'TempOperation','NomAgence',"Catégorisation du Temps d'Opération")
        st.altair_chart(chart2)
    elif option == "Temps Moyen & Type d'Opération":
        c1,c2=st.columns(2)
        
        plot_and_download(c1,GraphsGlob(df_all),'1')
        
        chart=stacked_service(df_all,type='NomService',concern='Type_Operation')
        c1.altair_chart(chart)
        
        plot_and_download(c2,Top10_Type(df_queue),'2')


    elif option == 'Agences Plus Lentes':
        c1,c2=st.columns([80,20])
        titre1="Top5 Agences les Plus Lentes en Temps d'Attente"
        c1.plotly_chart(area_graph(df_all,concern='NomAgence',time='TempsAttenteReel',date_to_bin='Date_Appel',seuil=15,title=titre1))
        c3,c4=st.columns([80,20])
        titre2="Top5 Agences les Plus Lentes en Temps de d'Operation"
        c3.plotly_chart(area_graph(df_all,concern='NomAgence',time='TempOperation',date_to_bin='Date_Fin',seuil=5,title=titre2))
    elif option == "Agences les plus Fréquentées":
        st.subheader("Top5 Agences les plus Fréquentées")
        c1,c2=st.columns(2)
        
        plot_and_download(c1,Top5Agence(df_all,df_queue,'Total Tickets','Nombre de Clients'),button_key='5')
        plot_and_download(c2,Top5Agence(df_all,df_queue,'Total Traités','Nombre de Clients'),button_key='6')
 
        
   


def option2(df_selected,df_queue):
    name_service=list(df_selected['NomService'].unique())
    st.subheader(f"{name_service}")
    
    

    pages = ['Top10 & Temps Moyen (Opération)','Agents & Queue',"Evolution Nbr Clients", "Performance Agents en Nbr de Clients", "Performance Agents en Temps"]
    
    option=st.radio(label=' ',options=pages,horizontal=True)
    
    
    if option == 'Top10 & Temps Moyen (Opération)':
        
        c1,c2=st.columns(2)
        plot_and_download(c1,Top10_Type_op(df_selected,df_all),button_key='7')
        plot_and_download(c2,TempsParType_op(df_selected,df_all),button_key='8')
        
    elif option== 'Agents & Queue':
        fig=stacked_chart(df_selected,'TempOperation','UserName',"Catégorisation du Temps d'opération")
        st.altair_chart(fig)    

    elif option == "Evolution Nbr Clients":
        
        c1,c2=st.columns([80,10])
        plot_and_download(c1,plot_line_chart(df_selected),button_key='9')
        
        
    
    elif option == "Performance Agents en Nbr de Clients":
        pie=Graphs_pie(df_selected)
        c4, c5,c6= st.columns([30,30,30])
        plot_and_download(c4,pie[0],button_key='p1')
        plot_and_download(c5,pie[1],button_key='p2')
        plot_and_download(c6,pie[2],button_key='p3')

    elif option == "Performance Agents en Temps":
        figs=Graphs_bar(df_selected)
        # Afficher les graphiques dans des colonnes
        c1, c2, c3 = st.columns([30,30,30])
        plot_and_download(c1,figs[0],button_key='f1')
        plot_and_download(c2,figs[1],button_key='f2')
        plot_and_download(c3,figs[2],button_key='f3')

    
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
    df=generate_agence(df,20)
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
            
            
