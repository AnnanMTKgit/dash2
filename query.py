import pyodbc
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import os
import random
server = st.secrets['db_server']
database = st.secrets['db_database']
username = st.secrets['db_username']
password = st.secrets['db_password']
driver = st.secrets['db_driver']



# Create the connection string
connection_string = (
    f'DRIVER={driver};'
    f'SERVER={server};'
    f'PORT=1433;'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password};'
    f'PWD={password};'
    # f'TRUSTED_CONNECTION=yes;'
)
connection = pyodbc.connect(connection_string)
# Agency Generating

def generate_agence(df):

    

    #new_agencies = [f'Agence_{i+1}' for i in range(num_new_agencies)]
    d={'Agence Ecobank Ngor':[14.754164, -17.506642],'Ecobank Yoff':[14.759643, -17.467951],

    'Ecobank Sacré Coeur':[14.723065, -17.469012],'Ecobank Maristes':[14.745871, -17.430285],

    'Ecobank Agence Ouakam':[14.724443, -17.487083],'Ecobank de Tilène':[14.683164, -17.446533],

    'Ecobank':[14.724788, -17.442350],'Ecobank Keur Massar':[14.781027, -17.319195],'ECOBANK':[14.708073, -17.437760],

    'Ecobank Hlm':[14.713197, -17.444832],'Ecobank Sénégal':[14.704719, -17.470716],'Ecobank Plateau':[14.672557, -17.432282],

    'EcoBank Sicap':[14.712269, -17.460110],'Ecobank Domaine Industriel':[14.725120, -17.442350]

    }
    
    
    # Update the 'NomAgence' column with new agency names in blocks of 500 rows
    for i,agency in enumerate(d.keys()):
        rows_per_agency = random.randint(0, int(len(df)/len(d.keys())))
        start_index = i * rows_per_agency
        end_index = start_index + rows_per_agency
        lat,long=d[agency]
        df.loc[start_index:end_index, 'NomAgence'] = agency
        df.loc[start_index:end_index, 'UserName'] = df['UserName'].apply(lambda x: f"{x}_{i+1}" if pd.notna(x) else x)
        df.loc[start_index:end_index, 'Longitude'] = long
        df.loc[start_index:end_index, 'Latitude'] =lat
    return df 

# Establish the connection


def sql_query(sql_requete,start_date, end_date):
   
    try:
        connection = pyodbc.connect(connection_string)
        df = pd.read_sql_query(sql_requete, connection,params=(start_date, end_date), index_col=None)
    except pyodbc.Error as pe:
        st.error(f"Error connecting to the database: {pe}")
        df = None
    finally:
        connection.close()
    # if multiAgence:
    #     df=generate_agence(df)
    return  df

@st.cache_data(hash_funcs={pyodbc.Connection: id}, show_spinner=False)
def sql_query_cached(_func,sql_requete,start_date, end_date):
    return _func(sql_requete,start_date, end_date)

def get_sqlData(sql_requete,start_date, end_date):
    current_date = datetime.now().date()
    current_hour=datetime.now().hour
        # Vérifier si les dates sont égales à la date actuelle
    if end_date == current_date and current_hour<18 and current_hour>7:
        # Forcer le rafraîchissement du cache en utilisant une clé spéciale
        
        df = sql_query(sql_requete,start_date, end_date)
        
    else:
        # Charger les données en utilisant la fonction de mise en cache normale
        df = sql_query_cached(sql_query,sql_requete,start_date, end_date)
    
    return  df

@st.cache_data(hash_funcs={pyodbc.Connection: id}, show_spinner=False)
def get_profil(sql_requete):
    try:
        connection = pyodbc.connect(connection_string)
        df = pd.read_sql_query(sql_requete, connection, index_col=None)
    except pyodbc.Error as pe:
        st.error(f"Error connecting to the database: {pe}")
        df = None
    finally:
        connection.close()
    
    return  df
