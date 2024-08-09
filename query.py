import pyodbc
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Define the connection parameters
server = 'marlodj-ecobank-db-server.database.windows.net'
database = 'MarlodjCore'
username = 'marlodj-admin'
password = 'tR0i48L658jQ'
driver = '{ODBC Driver 17 for SQL Server}'
#MARS_Connection='yes'
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



# Establish the connection



@st.cache_data(hash_funcs={pyodbc.Connection: id}, show_spinner=False)
def get_sqlData_cache(start_date, end_date):
   
    sql=f""" SELECT u.FirstName,u.LastName,u.UserName,q.Date_Reservation,q.Date_Appel,q.TempAttenteMoyen,DATEDIFF(second, q.Date_Reservation, q.Date_Appel) as TempsAttenteReel,
    q.Date_Fin,DATEDIFF(second, q.Date_Appel, q.Date_Fin) as TempOperation,q.IsMobile,e.Nom ,s.NomService,r.TypeOperationId,
    r.AgenceId,a.NomAgence,a.Capacites,a.Longitude,a.Latitude ,a.HeureFermeture FROM reservation r LEFT JOIN queue q ON r.id = q.reservationId
    LEFT JOIN Service s ON r.ServiceId = s.Id LEFT JOIN [User] u ON u.Id = q.userId LEFT JOIN Etat e ON e.Id = q.EtatId LEFT JOIN Agence a ON a.Id = r.AgenceId
WHERE Date_Reservation is not NULL and CAST(q.Date_Reservation AS DATE) BETWEEN CAST(? AS datetime) AND CAST(? AS datetime) 
ORDER BY q.Date_Reservation DESC;
  """
    try:
        connection = pyodbc.connect(connection_string)
        df = pd.read_sql_query(sql, connection,params=(start_date, end_date), index_col=None)
    except pyodbc.Error as pe:
        print("Error:", pe)
        df = None
    finally:
        connection.close()
    return  df


def get_sqlData(start_date, end_date):
    
    sql=f""" SELECT u.FirstName,u.LastName,u.UserName,q.Date_Reservation,q.Date_Appel,q.TempAttenteMoyen,DATEDIFF(second, q.Date_Reservation, q.Date_Appel) as TempsAttenteReel,
    q.Date_Fin,DATEDIFF(second, q.Date_Appel, q.Date_Fin) as TempOperation,q.IsMobile,e.Nom ,s.NomService,r.TypeOperationId,r.AgenceId,a.NomAgence,a.Capacites,a.Longitude,
    a.Latitude,a.HeureFermeture  FROM reservation r LEFT JOIN queue q ON r.id = q.reservationId LEFT JOIN Service s ON r.ServiceId = s.Id
    LEFT JOIN [User] u ON u.Id = q.userId LEFT JOIN Etat e ON e.Id = q.EtatId LEFT JOIN Agence a ON a.Id = r.AgenceId
WHERE Date_Reservation is not NULL and CAST(q.Date_Reservation AS DATE) BETWEEN CAST(? AS datetime) AND CAST(? AS datetime) 
ORDER BY q.Date_Reservation DESC;
  """
    try:
        connection = pyodbc.connect(connection_string)
        df = pd.read_sql_query(sql, connection,params=(start_date, end_date), index_col=None)
    except pyodbc.Error as pe:
        print("Error:", pe)
        df = None
    finally:
        connection.close()
    
    
    return  df