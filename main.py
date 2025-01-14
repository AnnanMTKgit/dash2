import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
import base64
from query import *
from sql import *
# Définir une liste d'utilisateurs et mots de passe valides



# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None

df_users=get_profil(SQLQueries().ProfilQueries)
    
users = dict(zip(df_users['UserName'], df_users['MotDePasse']))

# Function to show the login page
def show_login_page():
    
    st.sidebar.title("Connexion à Marlodj Dashboard")

    # Selectbox for username
    username = st.sidebar.text_input("Nom utilisateur")

    # Input for password
    password = st.sidebar.text_input("Mot de passe", type="password") 

    # Button for login
    if st.sidebar.button("Login"):
        # Validate the credentials
        if users.get(username) == password:
            #st.success(f"Welcome, {username}!")
            # Update session state to mark the user as logged in
            st.session_state.logged_in = True
            st.session_state.username = username
            # Rerun the app to move to the dashboard
            st.experimental_rerun()
            
            return username 
        else:
            st.error("Invalid username or password. Please try again.")

def show_dashboard_page(username):
    
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
    # Center the image using HTML
    placeholder = st.sidebar.empty()

    # Use a container with columns to center the image
    with placeholder.container():
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])  # Adjust the proportions as needed
        with col2:
            st.image('logo.png', caption="", width=100)
    
    
    def option_agent(df_all_service,df_queue_service):
        df=df_all_service.copy()
        nom=df["LastName"].iloc[0]
        prenom=df['FirstName'].iloc[0]
        nom_service=df["NomService"].iloc[0]
        
        st.sidebar.markdown(f'SERVICE : :orange[ {nom_service}]')
        #st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
        st.sidebar.markdown(f'UTILISATEUR : :blue[  {prenom} {nom}]')
        #st.sidebar.markdown(f"## Utilisateur :  {prenom} {nom}")
        st.sidebar.markdown("<br><br>", unsafe_allow_html=True) 
        # CSS styling
        st.markdown("""
<style>



[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

        col = st.columns((1.25, 5.25, 1), gap='medium')
        with col[0]:
            
            st.markdown(
    f"""
    <div style="text-align: center;">
        <p style="font-size: 20px; font-weight: bold;text-decoration: underline;">Totaux Opération</p>
    </div>
    """,
    unsafe_allow_html=True
)

            

            plot_metrics(df,'Traitée',"NombreTickets")
            plot_metrics(df,"Passée","NombreTickets")
            plot_metrics(df,"Rejetée","NombreTickets")

            
           

        with col[1]:
            c= st.columns((1.5, 1.5), gap='medium')
            with c[0]:
                fig=stacked_chart(df_all_service,'TempOperation','UserName',"Catégorisation du Temps d'opération")
                st.altair_chart(fig, use_container_width=True)  
            with c[1]:
                fig1=stacked_agent(df_all_service,type='UserName',titre="Nombre de type d'opération",concern='Type_Operation')
                st.altair_chart(fig1, use_container_width=True)
            
        with col[2]:
            st.markdown(
    f"""
    <div style="text-align: center;">
        <p style="font-size: 20px; font-weight: bold;text-decoration: underline;">File d'Attente</p>
    </div>
    """,
    unsafe_allow_html=True
)

            
           
            fig=service_congestion(df_queue_service,color=['#12783D','#00CC96'])
            st.altair_chart(fig,use_container_width=True)
            
            
        col = st.columns((1.25, 5.25, 1), gap='medium')
        
        with col[0]:
            st.markdown(
    f"""
    <div style="text-align: center;">
        <p style="font-size: 20px; font-weight: bold;text-decoration: underline;">Temps Moy Opération (MINUTES)</p>
    </div>
    """,
    unsafe_allow_html=True
)

            
            
            agg=ServiceTable(df,"Rejetée")
            
            plot_metrics(df,'Traitée',"TMO")
            plot_metrics(df,"Passée","TMO")
            plot_metrics(df,"Rejetée","TMO")
        with col[1]:
            st.plotly_chart(area_graph(df_all_service,concern='UserName',time='TempOperation',date_to_bin='Date_Fin',seuil=5,title="Evolution du temps moyen de traitement",couleur='#17becf'), use_container_width=True)
        

        with col[2]:
            fig1=service_congestion(df_queue,color=['#B03A30',"#EF553B"],title='Agence')
            st.altair_chart(fig1,use_container_width=True)

    def option1(df_all,df_queue,df_RH):
        page1='***Congestion et localisation***'
        page2='***Tableau Global***'
        page3='***Agences & Queue***'
        page4="***Temps Moyen & Type d'Opération***"
        page5='***Agences Plus Lentes***'
        page6="***Agences Plus Fréquentées***"
        page7="***Supervision des Agences***"
        pages = [page1,page2,page3,page4,page5 ,page6,page7]
        
        

        if "selected_page" not in st.session_state:
            st.session_state.selected_page = page1 # Par défaut, Page 1 est sélectionnée
        
        with open("stylebutton.css", "r") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        # Créer trois colonnes pour les boutons
        col1, col2, col3,col4,col5,col6,col7 = st.columns(7)

        with col1:
            if st.button(page1, help='Afficher les données sur la congestion et la localisation.',use_container_width=True):
                st.session_state.selected_page = page1
        with col2:
            if st.button(page2, help='Résumé global des données.',use_container_width=True):
                st.session_state.selected_page =page2
        with col3:
            if st.button(page3, help="Analyse des agences et files d'attente.",use_container_width=True):
                st.session_state.selected_page = page3

        with col4:
            if st.button(page4, help="Statistiques des temps moyens par type d'opération.",use_container_width=True):
                st.session_state.selected_page = page4
        with col5:
            if st.button(page5, help="Classement des agences avec le temps d'opération le plus lent.",use_container_width=True):
                st.session_state.selected_page = page5
        with col6:
            if st.button(page6, help="Les agences ayant le plus grand nombre de visites.",use_container_width=True):
                st.session_state.selected_page = page6
        with col7:
            if st.button(page7, help="Outil pour surveiller les performances des agences.",use_container_width=True):
                st.session_state.selected_page = page7
        # Afficher le contenu de la page sélectionnée
        if st.session_state.selected_page == page1:
            Conjection(df_all,df_queue)
        elif st.session_state.selected_page ==page2:
            HomeGlob(df_all,df_queue)
        elif st.session_state.selected_page == page3:
            col=st.columns((0.25, 4, 0.25), gap='medium')
            with col[1]:
                chart1=stacked_chart(df_all,'TempsAttenteReel','NomAgence',"Catégorisation du Temps d'Attente")
                st.altair_chart(chart1,use_container_width=True)
                chart2=stacked_chart(df_all,'TempOperation','NomAgence',"Catégorisation du Temps d'Opération")
                st.altair_chart(chart2,use_container_width=True)
                chart3=TempsPassage(df_all)
                st.altair_chart(chart3,use_container_width=True)

        elif st.session_state.selected_page == page4:
            col=st.columns([50,50], gap='medium')
            
            plot_and_download(col[0],GraphsGlob(df_all),'1')
            
            chart=stacked_service(df_all,type='NomService',concern='Type_Operation')
            with col[0]:
                st.altair_chart(chart,use_container_width=True)
            
            plot_and_download(col[1],Top10_Type(df_queue),'2')

        elif st.session_state.selected_page == page5:
            col=st.columns((0.25, 4, 0.25), gap='medium')
            
            with col[1]:
                titre1="Top5 Agences les Plus Lentes en Temps d'Attente"
                f,_,_,_=area_graph(df_all,concern='NomAgence',time='TempsAttenteReel',date_to_bin='Date_Appel',seuil=15,title=titre1)
                st.plotly_chart(f,use_container_width=True)
            
                titre2="Top5 Agences les Plus Lentes en Temps d'Operation"
                f,_,_,_=area_graph(df_all,concern='NomAgence',time='TempOperation',date_to_bin='Date_Fin',seuil=5,title=titre2)
                st.plotly_chart(f,use_container_width=True)  

        elif st.session_state.selected_page == page6:
            st.markdown(f"<h2 style='color:white; font-size:20px;text-align:center;'>Top5 Agences les plus Fréquentées</h2>", unsafe_allow_html=True)
            col=st.columns([50,50], gap='medium')
            
            with col[0]:
                st.plotly_chart(top_agence_freq(df_all,df_queue,title=['Total Tickets','Total Traités'],color=['#00CC96','#FFA15A']),use_container_width=True)
                st.plotly_chart(top_agence_freq(df_all,df_queue,title=['Total Tickets','Total Rejetées'],color=['#00CC96',"#EF553B"]),use_container_width=True)
            with col[1]:
                st.plotly_chart(top_agence_freq(df_all,df_queue,title=['Total Tickets','Total Passées'],color=['#00CC96','orange']),use_container_width=True)
        
        elif st.session_state.selected_page == page7:
             # Define the sub-pages
            subpage1 = 'Monitoring Congestion'
            subpage2 = 'Opération sur Rendez-vous'
            subpage3 = 'Opérations mises en attente'
            subpage4= "Evolution des temps dans la journée"
            sub_pages = [subpage1, subpage2, subpage3, subpage4]

            if "selected_page" not in st.session_state:
                st.session_state.selected_page = subpage1 # Par défaut, Page 1 est sélectionnée
        

            st.markdown('<div class="button-container">', unsafe_allow_html=True)
            # Créer trois colonnes pour les boutons
            col1, col2, col3,col4 = st.columns(4)
            
            with col1:
                if st.button(subpage1, help="Cliquez pour voir le contenu de la page 1"):
                    st.session_state.selected_page = subpage1
            with col2:
                if st.button(subpage2, help="Cliquez pour voir le contenu de la page 2"):
                    st.session_state.selected_page =subpage2
            with col3:
                if st.button(subpage3, help="Cliquez pour voir le contenu de la page 3"):
                    st.session_state.selected_page = subpage3
            with col4:
                if st.button(subpage4, help="Cliquez pour voir le contenu de la page 4"):
                    st.session_state.selected_page = subpage4


            # Switch between the sub-pages
            if st.session_state.selected_page == subpage1:
                # Call your congestion-related function here
                gird_congestion(df_all, df_queue)

            elif st.session_state.selected_page == subpage2:
                st.subheader("Pas de données")
                #point_rendez_vous(df_RH)
                

            elif st.session_state.selected_page == subpage3:
                st.subheader("Pas de données")
            
            elif st.session_state.selected_page == subpage4:
                col=st.columns((0.25, 4, 0.25), gap='medium')
            
                with col[1]:
                    titre1="Evolution du Temps d'Attente par Agence"
                    f,_,_,_=area_graph(df_all,concern='NomAgence',time='TempsAttenteReel',date_to_bin='Date_Appel',seuil=15,title=titre1)
                    st.plotly_chart(f,use_container_width=True)
                
                    titre2="Evolution du Temps d'Operation par Agence"
                    f,_,_,_=area_graph(df_all,concern='NomAgence',time='TempOperation',date_to_bin='Date_Fin',seuil=5,title=titre2)






       
    


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
        
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = page1 # Par défaut, Page 1 est sélectionnée
        
        with open("stylebutton.css", "r") as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        # Créer trois colonnes pour les boutons
        col1, col2, col3,col4,col5 = st.columns(5)

        with col1:
            if st.button(page1, help="Affiche les top 10 des opérations avec leur temps moyen.",use_container_width=True):
                st.session_state.selected_page = page1
        with col2:
            if st.button(page2, help="Analyse des agents et des files d'attente.",use_container_width=True):
                st.session_state.selected_page =page2
        with col3:
            if st.button(page3, help="Affiche l'évolution du nombre de clients au fil du temps.",use_container_width=True):
                st.session_state.selected_page = page3

        with col4:
            if st.button(page4, help="Analyse la performance des agents en fonction du nombre de clients.",use_container_width=True):
                st.session_state.selected_page = page4
        with col5:
            if st.button(page5, help="Analyse la performance des agents selon le temps qu'ils prennent pour traiter les clients.",use_container_width=True):
                st.session_state.selected_page = page5

        
       
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        

        # if option == page1:
            
        #     gird_congestion_service(df_selected,df_queue)

            
            
        
        if st.session_state.selected_page == page1:
            
            c1,c2=st.columns(2)
            plot_and_download(c1,Top10_Type_op(df_selected,df_all),button_key='7')
            plot_and_download(c2,TempsParType_op(df_selected,df_all),button_key='8')
            
        elif st.session_state.selected_page== page2:
            col=st.columns((0.25, 4, 0.25), gap='medium')
            with col[1]:
                fig=stacked_chart(df_selected,'TempOperation','UserName',"Catégorisation du Temps d'opération")
                st.altair_chart(fig,use_container_width=True)    
                fig1=stacked_agent(df_selected,type='UserName',concern='Type_Operation')
                st.altair_chart(fig1,use_container_width=True)
        elif st.session_state.selected_page == page3:
            
            col=st.columns((0.25, 4, 0.25), gap='medium')
            
            plot_and_download(col[1],plot_line_chart(df_selected),button_key='9')
            
            
        
        elif st.session_state.selected_page == page4:
            pie=Graphs_pie(df_selected)
            c4, c5,c6= st.columns([30,30,30])
            plot_and_download(c4,pie[0],button_key='p1')
            plot_and_download(c5,pie[1],button_key='p2')
            plot_and_download(c6,pie[2],button_key='p3')

        elif st.session_state.selected_page == page5:
            figs=Graphs_bar(df_selected)
            # Afficher les graphiques dans des colonnes
            c1, c2, c3 = st.columns([30,30,30])
            plot_and_download(c1,figs[0],button_key='f1')
            plot_and_download(c2,figs[1],button_key='f2')
            plot_and_download(c3,figs[2],button_key='f3')



    # Add a logout button
    if st.sidebar.button("Déconnexion"):
        # Clear session state and rerun the app to go back to login page
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()
    start_date, end_date=date_range_selection()

    df=get_sqlData(SQLQueries().AllQueueQueries,start_date, end_date)
        
    df_RH=get_sqlData(SQLQueries().RendezVousQueries,start_date,end_date)

    df_agences=get_profil(SQLQueries().AllAgences)

    if len(df)==0:
        
        st.warning('Données non disponible pour les Dates Choisies. Veuillez choisir des dates valides')
        st.warning("C'est peut être le Week-End ou un jour non ouvrable pour la banque")
        st.stop()
    else:
        
        username=st.session_state.username
        profil=df_users[df_users['UserName']==username]['Profil'].values[0]
        
        if profil in ['Caissier','Clientele']:
            df_queue=df.copy()
            df_all_service=df.query('UserName==@username')
           
            
            if df_all_service.empty:
                st.warning("Agent absent")
                st.stop()
            else:
                service=df_all_service["NomService"].iloc[0]
            
                df_queue_service=df.query('NomService==@service')
                option_agent(df_all_service,df_queue_service)
                
            if st.sidebar.button("Déconnexion"):
                # Clear session state and rerun the app to go back to login page
                st.session_state.logged_in = False
                st.session_state.username = None
                st.experimental_rerun()
            
        else:
            
            if profil in ['Administrateur','SuperAdmin']:
               
                
                NomAgence=st.sidebar.multiselect(
                    'Agences',
                    options=df_agences['NomAgence'].unique(),
                    default=df_agences['NomAgence'].unique()
                )
                df=df.query('NomAgence==@NomAgence')
                
                if df.empty:
                    st.warning("Pas de données pour cette sélection d'Agence dans cette période")
                    st.stop()
                
            else :
                NomAgence=df_users[df_users['UserName']==username]['NomAgence'].values[0]
                df=df.query('NomAgence==@NomAgence')
                st.sidebar.write(NomAgence)
            if df.empty:
                st.warning('Aucune Agence disponible')
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
                option1(df_all,df_queue,df_RH)
            

            if selected=="Filtre": # si la somme du Nbs de chaque service != Nbs Total alors UserId n'existe pas
                print('Waiting')    
                df_selected=filter1(df_all)
                if df_selected.empty:
                    st.warning('Données Vides')
                else:
                    option2(df_selected,df_queue)

            



# Logic to display the appropriate page
if st.session_state.logged_in:
    # Show dashboard if the user is logged in
    show_dashboard_page(username)
else:
    # Show login page if the user is not logged in
    username=show_login_page()
              
