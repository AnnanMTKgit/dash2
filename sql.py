class SQLQueries:
    def __init__(self):
        self.AllQueueQueries=f""" SELECT u.FirstName,u.LastName,u.UserName,q.Date_Reservation,q.Date_Appel,q.TempAttenteMoyen,DATEDIFF(second, q.Date_Reservation, q.Date_Appel) as TempsAttenteReel,
    q.Date_Fin,DATEDIFF(second, q.Date_Appel, q.Date_Fin) as TempOperation,q.IsMobile,e.Nom ,s.NomService,t.Label as Type_Operation,r.ReservationParHeure,
    r.AgenceId,a.NomAgence,a.Capacites,a.Longitude,a.Latitude ,a.HeureFermeture FROM reservation r LEFT JOIN TypeOperation t ON t.Id=r.TypeOperationId LEFT JOIN queue q ON r.id = q.reservationId
    LEFT JOIN Service s ON r.ServiceId = s.Id LEFT JOIN [User] u ON u.Id = q.userId LEFT JOIN Etat e ON e.Id = q.EtatId LEFT JOIN Agence a ON a.Id = r.AgenceId
    WHERE Date_Reservation is not NULL and CAST(q.Date_Reservation AS DATE) BETWEEN CAST(? AS datetime) AND CAST(? AS datetime) 
    ORDER BY q.Date_Reservation DESC;
    """
        self.ProfilQueries=f"""SELECT
      U.[FirstName]
      ,U.[LastName]
      ,U.[PhoneNumber]
      ,U.[UserName]
      ,U.[Token] as MotDePasse
      ,R.Label as Profil
      ,a.NomAgence
  FROM [User] U  LEFT JOIN [Role] R ON U.[RoleId]= R.Id LEFT JOIN Agence a ON a.Id = U.AgenceId"""
       
        self.RendezVousQueries= f"""SELECT RPH.Id
      ,[HeureReservation]
      ,[NumeroHeureReservation]
      ,[HeureAppel]
      ,[Date_Fin]
      ,[Date_Appel]
      ,[ReservationID]
      ,[userId]
      ,[TempAttenteMoyen]
      ,e.Nom
      ,[IsMobile]
      
  FROM [dbo].[ReservationParHeure] RPH LEFT JOIN Etat e ON e.Id = RPH.EtatId
  WHERE HeureReservation is not NULL and CAST(HeureReservation AS DATE) BETWEEN CAST(? AS datetime) AND CAST(? AS datetime) 
  """ 
        
        self.AllAgences=f"""SELECT [Id]
      ,[NomAgence]
      ,[Adresse]
      ,[codeAgence]
      ,[Pays]
      ,[RegionId]
      ,[Longitude]
      ,[Latitude]
      ,[StructureID]
      ,[NbClientByDay]
      ,[Status]
      ,[Capacites]
      ,[Telephone]
      ,[HeureDemarrage]
      ,[HeureFermeture]
      ,[SuspensionActivite]
      ,[ActivationReservation]
      ,[nombreLimitReservation]
  FROM [dbo].[Agence]"""