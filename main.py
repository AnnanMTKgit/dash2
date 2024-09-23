import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
import base64
st.set_page_config(page_title='Dashboard',page_icon='📊',layout='wide')
#st.sidebar.subheader('BienTableau de Bord de Marlodj 🇸🇳')
st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 20px;">
        <h1>Tableau de Bord de Marlodj 🇸🇳</h1>
    </div>
    """,
    unsafe_allow_html=True
)     

# Display the image in the main area
st.sidebar.image('logo.png', caption="", width=150)

# Custom CSS to make the button red


# st.markdown("""
#     <style>
#     @media (prefers-color-scheme: dark) {
#         /* Dark mode styles */
#         body {
#             color: #E0E0E0;  /* Light gray text for dark theme */
#             background-color: #000000;
#         }
#     }

#     @media (prefers-color-scheme: light) {
#         /* Light mode styles */
#         body {
#             color: #000000;  /* Dark text for light theme */
#             background-color: #FFFFFF;
#         }
#     }
#     </style>
# """, unsafe_allow_html=True)



def option1(df_all,df_queue):
    page1='***Congestion et localisation***'
    page2='***Tableau Global***'
    page3='***Agences & Queue***'
    page4="***Temps Moyen & Type d'Opération***"
    page5='***Agences Plus Lentes***'
    page6="***Agences Plus Fréquentées***"
    page7="***Supervision des Agences***"
    pages = [page1,page2,page3,page4,page5 ,page6,page7]
    st.markdown(
    """
    <style>
    /* Style général pour les labels */
    div[role='radiogroup'] {
        display: flex;
        flex-direction: row;  /* Disposer les labels horizontalement */
        flex-wrap: nowrap;    /* Empêcher le retour à la ligne */
    }
    div[role='radiogroup'] label {
        display: flex;
        padding: 3px;
        border: 2px solid #444;
        border-radius: 5px;
        background-color: #2E2E2E;  /* Couleur de fond normale (gris) */
        transition: background-color 0.3s ease;
        white-space: normal;  /* Autoriser le retour à la ligne */
        
    }
    /* Style au survol des labels */
    div[role='radiogroup'] label:click {
        background-color: #a9a9a9;  /* Couleur au survol (gris foncé) */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    option=st.radio(label=' ',options=pages,horizontal=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Afficher le contenu en fonction de la sélection
    if option == page1:
        
        Conjection(df_all,df_queue)

    elif option ==page2:
        
        HomeGlob(df_all,df_queue)
    elif option==page3:
        chart1=stacked_chart(df_all,'TempsAttenteReel','NomAgence',"Catégorisation du Temps d'Attente")
        st.altair_chart(chart1)
        chart2=stacked_chart(df_all,'TempOperation','NomAgence',"Catégorisation du Temps d'Opération")
        st.altair_chart(chart2)
        chart3=TempsPassage(df_all)
        st.altair_chart(chart3)
    elif option == page4:
        c1,c2=st.columns(2)
        
        plot_and_download(c1,GraphsGlob(df_all),'1')
        
        chart=stacked_service(df_all,type='NomService',concern='Type_Operation')
        c1.altair_chart(chart)
        
        plot_and_download(c2,Top10_Type(df_queue),'2')


    elif option == page5:
        c1,c2=st.columns([80,20])
        titre1="Top5 Agences les Plus Lentes en Temps d'Attente"
        c1.plotly_chart(area_graph(df_all,concern='NomAgence',time='TempsAttenteReel',date_to_bin='Date_Appel',seuil=15,title=titre1))
        c3,c4=st.columns([80,20])
        titre2="Top5 Agences les Plus Lentes en Temps d'Operation"
        c3.plotly_chart(area_graph(df_all,concern='NomAgence',time='TempOperation',date_to_bin='Date_Fin',seuil=5,title=titre2))
    elif option == page6:
       
        st.markdown(f"<h2 style='color:white; font-size:20px;text-align:center;'>Top5 Agences les plus Fréquentées</h2>", unsafe_allow_html=True)
        c1,c2=st.columns(2)
        
        fig1=topAg(df_all,df_queue,title=['Total Tickets','Total Traités'],color=['#00CC96','#FFA15A'])
        fig2=topAg(df_all,df_queue,title=['Total Tickets','Total Rejetées'],color=['#00CC96',"#EF553B"])
        fig3=topAg(df_all,df_queue,title=['Total Tickets','Total Passées'],color=['#00CC96','orange'])
        c1.plotly_chart(fig1)
        c1.plotly_chart(fig2)
        c2.plotly_chart(fig3)
        # c2.plotly_chart(top(df_all,df_queue,'Total Traités'))
    elif option ==page7:
        gird_congestion(df_all,df_queue)

        
   


def option2(df_selected,df_queue):
    liste = list(df_selected['NomService'].unique())
    services_str = '-'.join(liste)
    st.markdown(f'## Service(s) : {services_str}')

  


    st.markdown('---')
    
    page1='***Top10 & Temps Moyen (Opération)***'
    page2='***Agents & Queue***'
    page3='***Evolution du Nombre de Clients***'
    page4="***Performance Agent en Nombre de Clients***"
    page5='***Performance Agent en Temps***'



    pages = [page1,page2,page3,page4,page5]
    # Injecter du CSS personnalisé
    st.markdown(
    """
    <style>
    /* Style général pour les labels */
    div[role='radiogroup'] {
        display: flex;
        flex-direction: row;  /* Disposer les labels horizontalement */
        flex-wrap: nowrap;    /* Empêcher le retour à la ligne */
    
    }
    div[role='radiogroup'] label {
        padding: 3px;
        border: 2px solid #444;
        border-radius: 5px;
        background-color: #2E2E2E;  /* Couleur de fond normale (gris) */
        transition: background-color 0.3s ease;
    }
    /* Style au survol des labels */
    div[role='radiogroup'] label:click {
        background-color: #a9a9a9;  /* Couleur au survol (gris foncé) */
    }
    </style>
    """,
    unsafe_allow_html=True
)
    option=st.radio(label='',options=pages)
    st.markdown("<br><br>", unsafe_allow_html=True)

    
    if option == page1:
        
        c1,c2=st.columns(2)
        plot_and_download(c1,Top10_Type_op(df_selected,df_all),button_key='7')
        plot_and_download(c2,TempsParType_op(df_selected,df_all),button_key='8')
        
    elif option== page2:
        fig=stacked_chart(df_selected,'TempOperation','UserName',"Catégorisation du Temps d'opération")
        st.altair_chart(fig)    
        fig1=stacked_agent(df_selected,type='UserName',concern='Type_Operation')
        st.altair_chart(fig1)
    elif option == page3:
        
        c1,c2=st.columns([80,10])
        plot_and_download(c1,plot_line_chart(df_selected),button_key='9')
        
        
    
    elif option == page4:
        pie=Graphs_pie(df_selected)
        c4, c5,c6= st.columns([30,30,30])
        plot_and_download(c4,pie[0],button_key='p1')
        plot_and_download(c5,pie[1],button_key='p2')
        plot_and_download(c6,pie[2],button_key='p3')

    elif option == page5:
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
    #df=generate_agence(df)
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
            menu_title="",
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
            
            
