import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from query import *
import warnings
warnings.filterwarnings('ignore')
from streamlit_option_menu import option_menu
import time
import plotly.graph_objects as go
from streamlit.components.v1 import html
import pydeck as pdk
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import plotly.io as pio
import copy
######################### Global Analysis ####################


def current_attente(df_queue,service=None,agence=None):
    current_date = datetime.now().date()
    current_datetime = datetime.now()

# Set the time to 6:00 PM on the same day
    six_pm_datetime = current_datetime.replace(hour=18, minute=0, second=0, microsecond=0)

    if current_datetime > six_pm_datetime:
        return 0
    else:
        var='En attente'
        if service==None:
            
            df = df_queue.query(f"(Nom==@var & Date_Reservation.dt.strftime('%Y-%m-%d') == '{current_date}')")

            #df=df_queue.query(f"((Date_Appel.isnull() & Date_Fin.isnull()) or Date_Fin.isnull()) & (Date_Reservation.dt.strftime('%Y-%m-%d') == '{current_date}')")
            number=len(df)
        else: 
            df=df_queue.query(f"(Nom==@var & Date_Reservation.dt.strftime('%Y-%m-%d') == '{current_date}' & NomService==@service")
            number=len(df)
        return number



def create_map(data):
    legend_html = ''  # Variable pour stocker la légende HTML
    
    # Calcul de la bounding box
    min_lat = data['Latitude'].min()
    max_lat = data['Latitude'].max()
    min_lon = data['Longitude'].min()
    max_lon = data['Longitude'].max()

    # Définition du polygone à partir de la bounding box
    polygon_coords = [
        [min_lon, min_lat],
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat]
    ]

    polygon_data = {
        'coordinates': [polygon_coords],
        'name': 'Bounding Box'
    }

    # Définir la vue initiale de la carte
    initial_view = pdk.ViewState(
        latitude=data['Latitude'].mean(),
        longitude=data['Longitude'].mean(),
        zoom=13,
        pitch=150
    )

    # Extraire tous les lieux uniques de la colonne 'place'
    unique_places = data['NomAgence'].unique()

    # Générer une palette de N couleurs en fonction du nombre de lieux uniques
    colors = cm.tab10.colors[:len(unique_places)]  # Utilisation d'une palette de couleurs de matplotlib

    # Créer un dictionnaire associant chaque lieu unique à une couleur de la palette
    color_scale = {place: colors[i] for i, place in enumerate(unique_places)}

    # Liste pour stocker les couches ColumnLayer
    layers = []

    # Ajouter une couche ColumnLayer pour chaque lieu avec sa couleur correspondante
    for place in unique_places:
        color_rgb = [int(c * 255) for c in color_scale[place]]  # Conversion RGB pour PyDeck
        layer = pdk.Layer(
            'ColumnLayer',
            data=data[data['NomAgence'] == place],
            get_position='[Longitude, Latitude]',
            get_elevation=100,  # Hauteur fixe pour tous les bâtiments
            elevation_scale=10,
            get_fill_color=color_rgb,  # Utilisation de la couleur RGB
            radius=50,
            pickable=True,
            extruded=True,
            id=place
        )
        layers.append(layer)

        # Ajouter à la légende HTML
        legend_html += f'<div><span style="color:rgb({", ".join(map(str, color_rgb))});">&#9632;</span> {place}</div>'

    # Définir la couche du polygone pour couvrir la zone
    polygon_layer = pdk.Layer(
        'PolygonLayer',
        data=[polygon_data],
        get_polygon='coordinates',
        get_fill_color=[0, 0, 0, 0],
        get_line_color=[0, 0, 255],
        line_width_min_pixels=2,
        pickable=True
    )

    # Définir la configuration de la carte avec un style
    deck = pdk.Deck(
        initial_view_state=initial_view,
        layers=layers + [polygon_layer],
        tooltip={"html": "<b>Nom:</b> {NomAgence} <br><b>Latitude:</b> {Latitude} <br><b>Longitude:</b> {Longitude}", "style": {"color": "white"}},
        #tooltip={"text": "Lieu: {place}\nLat: {latitude}\nLon: {longitude}"},  # Afficher le nom du lieu dans le tooltip
        map_style="mapbox://styles/mapbox/satellite-streets-v11",#'mapbox://styles/mapbox/streets-v11',  # Spécifier le style de la carte
        width=500,
        height=100
    )

    # Retourner la légende HTML
    return legend_html, deck




def plot_and_download(col, fig, button_key):
    # Configure the figure to be static
    config = {
        'staticPlot': True,  # Disable all interactive features
        'displayModeBar': False  # Hide the mode bar (but still offer the download button via Streamlit)
    }
   
    copied_fig = copy.deepcopy(fig)
    copied_fig.update_layout(margin=dict(l=200, r=100, t=50, b=50),
                             autosize=False,
    width=1200,  # Set figure width
    height=800 

      # Margins in pixels
)
    img_bytes = pio.to_image( copied_fig, format='png',scale=2)
    with col:
        # Add a download button
        col.download_button(
            label="⬇",
            data=img_bytes,
            file_name='plot.png',
            mime='image/png',
            key=button_key 
        ) 
        # Display the Plotly chart
        col.plotly_chart(fig,config=config,use_container_width=True)
        
        



def Conjection(df_queue):
    c1,c2,c3=st.columns([30,55,15])
    legend_html, deck =create_map(df_queue)
    NomAgence = c1.selectbox(
        ':white[CONGESTION PAR AGENCE:]',
        options=df_queue['NomAgence'].unique(),
        index=0,
        key='2'
    )
    
    df = df_queue.query('NomAgence==@NomAgence')
    
    
    with c2:
        c2.pydeck_chart(deck)
        
        
    with c3:
        c3.write('Legend')
        c3.markdown(legend_html, unsafe_allow_html=True)
    max_length=df['Capacites'].unique()[0]
    queue_length=current_attente(df)

    percentage = (queue_length / max_length) * 100
    
    display_value = queue_length if queue_length < max_length else " Capacité Atteinte"
    bar_color = 'red' if queue_length >= max_length else ('white' if percentage ==0 else
        'green' if percentage < 50 else 'yellow' if percentage < 80 else "orange"
    )
    titre={"white":'Vide','green':"Modérement occupée","yellow":"Fortement occupée","orange":"Très fortement occupée ",'red':'Congestionnée'}
    prefix_text = (
    f"<span style='color:white; font-size:20px;'>"
    f"Client(s) en Attente: <span style='color:{bar_color};'>{queue_length}</span>"
    "</span>"
)


    fig = go.Figure(go.Indicator(
        mode = "gauge+delta",
        value = queue_length,
        number={'suffix': '' if queue_length <= max_length else display_value},
        delta = {'reference': -1, 'increasing': {'color': "#000000"},'prefix': prefix_text ,'font': {'size': 24}},
        #delta={'reference': 0, 'increasing': {'color': bar_color}},
        gauge = {
            'axis': {'range': [0, max_length]},
            'bar': {'color': bar_color, 'thickness': 0.00002},  # Barre très fine
            'steps': [
                
                {'range': [0, 0.5 * max_length], 'color': 'green'},
                {'range': [0.5 * max_length, 0.80 * max_length], 'color': 'yellow'},
                {'range': [0.80 * max_length, max_length], 'color': "orange"},
                

            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.95,
                'value': queue_length
            }
        },
        title = {'text': titre[bar_color], 'font': {'size': 18, 'color': bar_color}},  # Increased font size to 24
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    if queue_length > max_length:
        fig.update_traces(number={'valueformat': "d", 'font': {'size': 12}, 'suffix': display_value})
    fig.update_layout(
        height=400,
        margin=dict(l=30, r=30, t=30, b=30),plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
    xaxis_title='Client(s) en Attente',  # Ajouter le titre de l'axe des X
    xaxis_title_font=dict(size=16, color='white'),  # Définir la taille et la couleur de la police
    )
    
    
    c1.plotly_chart(fig,use_container_width=True)

        
        
        


         
def GraphsGlob(df_all):
    

    df=(df_all.groupby(by=['NomService']).mean()['TempOperation']/60).astype(int).reset_index()
    

    fig_tempOp_1 = go.Figure()
   
    fig_tempOp_1.add_trace(go.Bar(go.Bar(x=df['NomService'],y=df['TempOperation'],orientation='v',text=df['TempOperation'],width=[0.6] * len(df['NomService']) , # Réduire la largeur de la barre
    textposition='inside',showlegend=False,marker=dict(color='#0083b8')
    )))


    fig_tempOp_1.update_layout(title=f"Temps Moyen d'Opération par Type de Service", xaxis=(dict(tickmode='linear')),width=400,
        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                             yaxis=(dict(title='Temps (min)',showgrid=False)))
     

    return fig_tempOp_1


def AgenceTable(df_all,df_queue):
    df1=df_all.copy()
    agg1 = df1.groupby(['NomAgence', 'Capacites']).agg(
    Temps_Moyen_Operation=('TempOperation', lambda x: np.round(np.mean(x)/60).astype(int)),
    Temps_Moyen_Attente=('TempsAttenteReel', lambda x: np.round(np.mean(x)/60).astype(int)),NombreTraites=('UserName',lambda x: int(sum(x.notna())))
).reset_index()
    
    df2=df_queue.copy()
    agg2=df2.groupby(['NomAgence', 'Capacites']).agg(NombreTickets=('Date_Reservation', np.count_nonzero),AttenteActuel=("NomAgence",lambda x: current_attente(df2,agence=x.values[0])),TotalMobile=('IsMobile',lambda x: int(sum(x)))).reset_index()
    agg=pd.merge(agg1,agg2,on=['NomAgence', 'Capacites'],how='left')
    agg=agg.rename(columns={'NomAgence':"Nom d'Agence",'Capacites':'Capacité','Temps_Moyen_Operation':"Temps Moyen d'Operation (MIN)",'Temps_Moyen_Attente':"Temps Moyen d'Attente (MIN)",'NombreTraites':'Total Traités','NombreTickets':'Total Tickets','AttenteActuel':'Nbs de Clients en Attente'})
    agg["Period"]=f"Du {df_queue['Date_Reservation'].min().strftime('%Y-%m-%d')} à {df_queue['Date_Reservation'].max().strftime('%Y-%m-%d')}"
    order=['Period',"Nom d'Agence", "Temps Moyen d'Operation (MIN)", "Temps Moyen d'Attente (MIN)",'Capacité','Total Tickets','Total Traités','TotalMobile','Nbs de Clients en Attente']
    agg=agg[order]
    
    return agg

def HomeGlob(df_all,df_queue):
    agg=AgenceTable(df_all,df_queue)
    cmap = plt.cm.get_cmap('RdYlGn')
    st.dataframe(agg.style.background_gradient(cmap=cmap,vmin=(-0.015),vmax=0.015,axis=None))
    #st.write(agg.style.background_gradient(cmap=cmap,vmin=(-0.015),vmax=0.015,axis=None).to_html(), unsafe_allow_html=True)
    
    


def Top10_Type(df_queue):
    df=df_queue.copy()
    df['TypeOperationId'] = df['TypeOperationId'].apply(lambda x: 0 if pd.isnull(x) else x)
    mapping={0:'Inconnu',1: 'Retrait_Espèce',2: 'Retrait_Chèque',3: 'Gros_Retrait',4: 'Retrait_carte',5: 'Demande_Chequier',6: 'Paiement_Frais_Visa_USA',7: 'Gros_Versement',8: 'Versement',9: 'Autre_dépôt',10: 'CGF_Bourse',11: 'Ciment_Sahel',12: ' Dangote',13: 'FT 2',14: 'Hapag_Lloyd',15: 'MSC',16: 'Total_Energies',17: 'Achat_devises',18: 'Vent_ devises',19: 'Moneygram',20: 'Rapid_Transfert',21: 'RIA',22: 'Transfast',23: 'Western_Union',24: 'Autr_ transfert'}
    df['TypeOperationId'] = df['TypeOperationId'].apply(lambda x: mapping[x])
    top_counts = df['TypeOperationId'].value_counts().reset_index()
    top_counts=top_counts.sort_values(by='TypeOperationId', ascending=False)
    top_counts=top_counts.head(10)
    top_counts = top_counts.iloc[::-1]
    
    
    fig = go.Figure()
    if top_counts.empty==False:
        valmax=top_counts['TypeOperationId'].max()
        
        dfmax=top_counts[top_counts['TypeOperationId'].apply(lambda x:(x>=100) and (valmax-x<=100))]
    
        dfmin=top_counts[top_counts['TypeOperationId'].apply(lambda x:(x<100) or (valmax-x>100))]
    # Ajouter les barres pour les valeurs < 100
        
        # Ajouter les barres pour les valeurs >= 100
        fig.add_trace(go.Bar(go.Bar(x=dfmin['TypeOperationId'], y=dfmin['index'],orientation='h',text=dfmin['TypeOperationId'],
        textposition='outside',showlegend=False,textfont=dict(color='white'),marker=dict(color='green'))
        ))
        fig.add_trace(go.Bar(go.Bar(x=dfmax['TypeOperationId'], y=dfmax['index'],orientation='h',text=dfmax['TypeOperationId'],
        textposition='inside',showlegend=False,textfont=dict(color='white'),marker=dict(color='green'))
        ))

    fig.update_layout(title=f"Top 10 Type d'Opération en nombre de clients",plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(title='Nombre de Clients',tickfont=dict(size=10)),width=600,margin=dict(l=150, r=50, t=50, b=150),
                  yaxis=dict(title='Type'))
    
    return fig

def Top5Agence(df_all,df_queue,title,text):
    agg=AgenceTable(df_all,df_queue)
    top_counts=agg.sort_values(by=title, ascending=False)
    top_counts=top_counts.head(5)
    top_counts = top_counts.iloc[::-1]
    
    
    fig = go.Figure()
    if top_counts.empty==False:
        valmax=top_counts[title].max()
        
        dfmax=top_counts[top_counts[title].apply(lambda x:(x>=100) and (valmax-x<=100))]
    
        dfmin=top_counts[top_counts[title].apply(lambda x:(x<100) or (valmax-x>100))]
    # Ajouter les barres pour les valeurs < 100
        
        # Ajouter les barres pour les valeurs >= 100
        fig.add_trace(go.Bar(go.Bar(x=dfmin[title], y=dfmin["Nom d'Agence"],orientation='h',text=dfmin[title],width=[0.6] * len(dfmin["Nom d'Agence"]) ,  # Réduire la largeur de la barre
        textposition='outside',showlegend=False,textfont=dict(color='white'),marker=dict(color= '#0083b8'))
        ))
        fig.add_trace(go.Bar(go.Bar(x=dfmax[title], y=dfmax["Nom d'Agence"],orientation='h',text=dfmax[title],width=[0.6] * len(dfmax["Nom d'Agence"]) ,  # Réduire la largeur de la barre
        textposition='inside',showlegend=False,textfont=dict(color='white'),marker=dict(color= '#0083b8'))
        ))
    
    fig.update_layout(title=f'{title}',plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(title=text,tickfont=dict(size=10)),width=500,margin=dict(l=150, r=50, t=50, b=150),
                  yaxis=dict(title="Nom d'Agence"))
    
    return fig




######################## Analysis with filter ###################

def Top10_Type_op(df_selection,df_all):
    df=df_all.copy()
    NomService=list(df_selection['NomService'].values)
    df=df[df['NomService'].isin(NomService)]
    df['TypeOperationId'] = df['TypeOperationId'].apply(lambda x: 0 if pd.isnull(x) else x)
    mapping={0:'Inconnu',1: 'Retrait_Espèce',2: 'Retrait_Chèque',3: 'Gros_Retrait',4: 'Retrait_carte',5: 'Demande_Chequier',6: 'Paiement_Frais_Visa_USA',7: 'Gros_Versement',8: 'Versement',9: 'Autre_dépôt',10: 'CGF_Bourse',11: 'Ciment_Sahel',12: ' Dangote',13: 'FT 2',14: 'Hapag_Lloyd',15: 'MSC',16: 'Total_Energies',17: 'Achat_devises',18: 'Vent_ devises',19: 'Moneygram',20: 'Rapid_Transfert',21: 'RIA',22: 'Transfast',23: 'Western_Union',24: 'Autr_ transfert'}
    df['TypeOperationId'] = df['TypeOperationId'].apply(lambda x: mapping[x])
    top_counts = df['TypeOperationId'].value_counts().reset_index()
    top_counts=top_counts.sort_values(by='TypeOperationId', ascending=False)
    top_counts=top_counts.head(10)
    top_counts = top_counts.iloc[::-1]
   
    fig = go.Figure()
    if top_counts.empty==False:
        valmax=top_counts['TypeOperationId'].max()
        
        dfmax=top_counts[top_counts['TypeOperationId'].apply(lambda x:(x>=100) and (valmax-x<=100))]
    
        dfmin=top_counts[top_counts['TypeOperationId'].apply(lambda x:(x<100) or (valmax-x>100))]
    
        fig.add_trace(go.Bar(go.Bar(x=dfmin['TypeOperationId'], y=dfmin['index'],orientation='h',text=dfmin['TypeOperationId'],
        textposition='outside',showlegend=False,marker=dict(color='#0083b8'))
        ))
        fig.add_trace(go.Bar(go.Bar(x=dfmax['TypeOperationId'], y=dfmax['index'],orientation='h',text=dfmax['TypeOperationId'],
        textposition='inside',showlegend=False,marker=dict(color='#0083b8'))
        ))

    fig.update_layout(title=f'Top 10 Type Opérations Traitées en Nombre de Clients ',plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(title='Nombre de Clients',tickfont=dict(size=10)),width=500,margin=dict(l=150, r=50, t=50, b=150),
                  yaxis=dict(title='Type'))
    return fig


def TempsParType_op(df_selection,df_all):
    df=df_all.copy()
    NomService=list(df_selection['NomService'].values)
    df=df[df['NomService'].isin(NomService)]
    df['TypeOperationId'] = df['TypeOperationId'].apply(lambda x: 0 if pd.isnull(x) else x)
    mapping={0:'Inconnu',1: 'Retrait_Espèce',2: 'Retrait_Chèque',3: 'Gros_Retrait',4: 'Retrait_carte',5: 'Demande_Chequier',6: 'Paiement_Frais_Visa_USA',7: 'Gros_Versement',8: 'Versement',9: 'Autre_dépôt',10: 'CGF_Bourse',11: 'Ciment_Sahel',12: ' Dangote',13: 'FT 2',14: 'Hapag_Lloyd',15: 'MSC',16: 'Total_Energies',17: 'Achat_devises',18: 'Vent_ devises',19: 'Moneygram',20: 'Rapid_Transfert',21: 'RIA',22: 'Transfast',23: 'Western_Union',24: 'Autr_ transfert'}
    df['TypeOperationId'] = df['TypeOperationId'].apply(lambda x: mapping[x])
    top_counts =df.groupby('TypeOperationId').agg(TempOperation=('TempOperation', lambda x:  np.round(np.mean(x)/60))).reset_index()
    top_counts=top_counts.sort_values(by='TempOperation', ascending=False)
    
    top_counts = top_counts.iloc[::-1]
    fig = go.Figure()
    if top_counts.empty==False:
    
        valmax=top_counts['TempOperation'].max()
        
        dfmax=top_counts[top_counts['TempOperation'].apply(lambda x:(x>=100) and (valmax-x<=100))]
    
        dfmin=top_counts[top_counts['TempOperation'].apply(lambda x:(x<100) or (valmax-x>100))]
    
        fig.add_trace(go.Bar(go.Bar(x=dfmin['TempOperation'], y=dfmin['TypeOperationId'],orientation='h',text=dfmin['TempOperation'],
        textposition='outside',showlegend=False,marker=dict(color='#0083b8'))
        ))
        fig.add_trace(go.Bar(go.Bar(x=dfmax['TempOperation'], y=dfmax['TypeOperationId'],orientation='h',text=dfmax['TempOperation'],
        textposition='inside',showlegend=False,marker=dict(color='#0083b8'))
        ))

    fig.update_layout(title=f"Temps Moyen Par Type d'Opérations Traitées ",plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
                  xaxis=dict(title='Temps Moyen en minutes',tickfont=dict(size=10)),width=500,margin=dict(l=150, r=50, t=50, b=150),
                  yaxis=dict(title='Type'))
    return fig
    


def filtering(df, UserName, NomService):
              
    return df.query('UserName in @UserName & NomService in @NomService')



def filter1(df_all):
    
    NomService = st.sidebar.multiselect(
        'Services',
        options=df_all['NomService'].unique(),
        default=df_all['NomService'].unique()
    )
    
    # Filter df_all based on the selected NomService
    df = df_all[df_all['NomService'].isin(NomService)]

    # UserName selection
    UserName = st.sidebar.multiselect(
        'Agents',
        options=df['UserName'].unique(),
        default=df['UserName'].unique()
    )
    
    df_selection = filtering(df, UserName, NomService)



    return df_selection



def create_bar_chart(df, status, title):
    
    df_filtered = df[df['Nom'] == status]
    top = df_filtered.groupby(by=['UserName']).agg(TempOperation=('TempOperation',lambda x: np.round(np.mean(x)/60))).reset_index()
    top=top.sort_values(by='TempOperation', ascending=True)

    fig = go.Figure()
    if top.empty==False:
        
        valmax=top['TempOperation'].max()
        
        dfmax=top[top['TempOperation'].apply(lambda x:(x>=10) and (valmax-x<=10))]
    
        dfmin=top[top['TempOperation'].apply(lambda x:(x<10) or (valmax-x>10))]
        
        fig.add_trace(go.Bar(go.Bar(x=dfmin['TempOperation'], y=dfmin['UserName'],orientation='h',text=dfmin['TempOperation'],
        textposition='outside',showlegend=False,marker=dict(color='#0083b8'))
        ))
        fig.add_trace(go.Bar(go.Bar(x=dfmax['TempOperation'], y=dfmax['UserName'],orientation='h',text=dfmax['TempOperation'],
        textposition='inside',showlegend=False,marker=dict(color='#0083b8'))
        ))

    
        
    fig.update_layout(
    title=f'Temps Moyen Opération {status}',
    xaxis_title='Temps en minutes',
    yaxis_title='Agents',
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="white"
    ),
    plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        showgrid=False
    
    ),
    yaxis=dict(
        showgrid=False,
    )
)   
    fig.update_traces(hovertemplate='%{label}: %{value}<extra></extra>')
    return fig
    

def create_pie_chart(df, title):

    df=df[df['Nom']==title]
    top = df.groupby(by=['UserName'])['Nom'].count().reset_index()
    

    fig = go.Figure()
   
    
    if top.empty==False:
        top['LabelWithNbs'] = top['UserName'] + ' (' + top['Nom'].round(2).astype(str) + ')'

        fig.add_trace(go.Pie(
            labels=top['LabelWithNbs'],
            values=top['Nom'],
            pull=[0.1 if i == 1 else 0 for i in range(len(top))],  # Pull out the second slice ('B')
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'], line=dict(color='#FFFFFF', width=2)),
            textinfo='percent' ,textposition= 'inside' 
        ))
        

    # Update layout for aesthetics
    fig.update_layout(
        title_text=f'Personnes {title}s Par Agent',
        legend=dict(
        title="Legend",
        itemsizing='constant',
        font=dict(size=10)
    ),
        annotations=[dict(text='', x=0.5, y=0.5, font_size=12, showarrow=False)],
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


    fig.update_traces(hovertemplate='%{label}: %{value}<extra></extra>')
    return fig
    


def ServiceTable(df_selected,df_queue):
    df1=df_selected.copy()
   
    agg1 = df1.groupby(['NomAgence']).agg(
    Temps_Moyen_Operation=('TempOperation', lambda x: np.round(np.mean(x)/60)),
    Temps_Moyen_Attente=('TempsAttenteReel', lambda x: np.round(np.mean(x)/60)),NombreTraites=('UserName',lambda x: int(sum(x.notna())))
).reset_index()
    
    df2=df_queue.copy()
    
    agg2=df2.groupby(['NomAgence']).agg(NombreTickets=('Date_Reservation', np.count_nonzero),AttenteActuel=("NomAgence",lambda x: current_attente(df2,agence=x.values[0])),TotalMobile=('IsMobile',lambda x: int(sum(x)))).reset_index()
    agg=pd.merge(agg1,agg2,on=['NomAgence'],how='left')
    agg=agg.rename(columns={'NomAgence':"Nom d'Agence",'Temps_Moyen_Operation':"Temps Moyen d'Operation (MIN)",'Temps_Moyen_Attente':"Temps Moyen d'Attente (MIN)",'NombreTraites':'Total Traités','NombreTickets':'Total Tickets','AttenteActuel':'Nbs de Clients en Attente'})
    agg["Period"]=f"Du {df_queue['Date_Reservation'].min().strftime('%Y-%m-%d')} à {df_queue['Date_Reservation'].max().strftime('%Y-%m-%d')}"

    order=['Period',"Nom d'Agence", "Temps Moyen d'Operation (MIN)", "Temps Moyen d'Attente (MIN)",'Total Tickets','Total Traités','TotalMobile','Nbs de Clients en Attente']
    agg=agg[order]
    
    return agg

def HomeFilter(df_selected,df_queue):
    
    agg=ServiceTable(df_selected,df_queue)
    
        
    return agg
def Graphs_bar(df_selected):
    
    
    figs = [
        create_bar_chart(df_selected, 'Traitée', 'TempOperation/Traitée'),
        #create_bar_chart(df_selection, 'En Attente', 'En attente'),
        create_bar_chart(df_selected, 'Passée', 'TempOperation/Passée'),
        create_bar_chart(df_selected, 'Rejetée', 'TempOperation/Rejetée')
    ]
    config = {
    'staticPlot': True,  # Disable all interactive features
    'displayModeBar': False  # Hide the mode bar (but still offer the download button via Streamlit)
}
    return figs
    
    

def Graphs_pie(df_selected):
    pie=[
        create_pie_chart(df_selected, 'Traitée'),
        #create_pie_chart(df_selection, 'En attente'),
        create_pie_chart(df_selected, 'Passée'),
        create_pie_chart(df_selected, 'Rejetée')
    ]
    return pie
    

# Plotting with Plotly
def find_highest_peak(df, person):
        df_person = df[df['UserName'] == person]
        max_row = df_person.loc[df_person['count'].idxmax()]
        return max_row['Date_Reservation']
def find_value_peak(df, person):
        df_person = df[df['UserName'] == person]
        return df_person['count'].max()

def plot_line_chart(df):
    if len(df['Date_Reservation'].dt.date.unique())==1:

        grouped = df.groupby('UserName').size().reset_index(name='count')
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=grouped['UserName'],
            y=grouped['count'],
            mode='lines+markers+text',
            text=grouped.apply(lambda row: f"( {row['count']})", axis=1),
            textposition='top center',
            marker=dict(size=9),
            name='Total Count'
            

        ))

        fig.update_layout(
            xaxis=dict(title='Agent',tickfont=dict(size=10)),
            yaxis=dict(title="Nombre d'Opération"),margin=dict(l=150, r=20, t=20, b=150),
            title="Nombre d'Opération par Agent",plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,height=500
        )
        
    
    else:

        df['date'] = df['Date_Reservation'].dt.date

        filtered_df = df

        # Agréger les données par jour et par personne
        aggregated_df = filtered_df.groupby(['UserName', 'date']).size().reset_index(name='count')
        
        aggregated_df['Date_Reservation'] = aggregated_df['UserName'] + ' = ' + aggregated_df['date'].astype(str)
       
        # Récupération des dates des pics les plus élevés pour chaque personne
        peak_dates = {person: find_highest_peak(aggregated_df, person) for person in aggregated_df['UserName'].unique()}

        # Filtrage des dates d'abscisse pour n'afficher que les dates des pics
        peak_date_strings = [date for date in peak_dates.values()]
        
        agg=aggregated_df.loc[aggregated_df.groupby('UserName')['count'].idxmax()]

        # Créer le graphique
        
        fig = px.line(aggregated_df, x='Date_Reservation', y='count', color='UserName',line_group='UserName', title='Nombre d\'Opération par Agent', markers=True)
        fig.update_xaxes(
        tickmode='array',
        tickvals=[date for date in peak_dates.values()],
        ticktext=peak_date_strings
       )  
    
        fig.add_trace(go.Scatter(
            x=agg['Date_Reservation'],
            y=agg['count'],
            mode='text',
            text=agg.apply(lambda row: f" {row['count']}", axis=1),
            textposition='top center',
            marker=dict(size=10),
            showlegend=False

        ))
        

        fig.update_layout(
            xaxis_title='Date de Pick de Client par Agent',
            yaxis_title='Nombre d\'Opérations',
            xaxis_tickangle=-45,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            # xaxis=dict(
            #     tickmode='array',
            #     tickvals= aggregated_df['Date_Reservation'],
            #     ticktext=[pd.to_datetime(d).strftime('%d-%m-%y') for d in aggregated_df['date']]
            # ),
            height=500

        )
    return fig  



def date_range_selection():
    start_date = st.sidebar.date_input(":white[Date Début]")
    end_date = st.sidebar.date_input(":white[Date Fin]")

    if start_date <= end_date:
        #st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
        return start_date, end_date
    else:
        st.warning('Mauvaise Date')
        st.stop()
        



 
 





