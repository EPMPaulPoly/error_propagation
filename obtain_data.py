import pandas as pd
from sqlalchemy import create_engine
import os
import dotenv as de
import numpy as np
    

def obtain_data():
    eng = get_connection()
    with eng.connect() as con1:
        query = f'''
            SELECT 
                * 
            from
                assignation_strates
        '''
        strata_assignments = pd.read_sql(query,con=con1)
        lots = strata_assignments['g_no_lot'].unique().tolist()
        query_obs = f'''
                SELECT 
                    g_no_lot,
                    n_places as y_obs,
                    id_strate
                FROM
                    resultats_validation
            '''
        obs_data:pd.DataFrame = pd.read_sql(query_obs,con=con1)
        lots = obs_data['g_no_lot'].unique().tolist()
        pred_data_query = f'''
            SELECT 
                g_no_lot,
                CEIL(n_places_min)::int as y_pred
            from inventaire_stationnement
            where g_no_lot in ('{"','".join(lots)}') and methode_estime=2
        '''
        pred_data = pd.read_sql(pred_data_query,con=con1)
        data_out = obs_data.merge(pred_data,how='left',on='g_no_lot')
    return data_out
    
def obtain_strata():
    eng = get_connection()
    with eng.connect() as con1:
        query = f'''
            SELECT 
                * 
            from
                conditions_strates_a_echant
        '''
        strata_assignments = pd.read_sql(query,con=con1)
    return strata_assignments

def obtain_population_sizes():
    eng = get_connection()
    with eng.connect() as con1:
        query = '''SELECT 
                        ass.id_strate::int,
                        csae.desc_concat,
                        count(*)::int as popu_strate
                    from inputs_validation iv
                    left join association_strates ass on ass.g_no_lot = iv.g_no_lot
                        left join conditions_strates_a_echant csae on csae.id_strate = ass.id_strate
                    group by ass.id_strate,csae.desc_concat'''
        comptes = pd.read_sql(query,con=con1)
    return comptes
def obtain_parking_estimate_strata(id_strate:int):
    eng = get_connection()
    with eng.connect() as con1:
        lots_query = f''' 
            SELECT 
                g_no_lot
            from association_strates 
            where id_strate ={id_strate}
        '''
        lots = pd.read_sql(lots_query,con=con1)
        lots_list = lots['g_no_lot'].unique().tolist()
        lots_parking = f'''
            SELECT
                sum(CEIL(n_places_min))::int as y_tot
            from inventaire_stationnement
            where g_no_lot in ('{"','".join(lots_list)}') and methode_estime=2
        '''
        lots_parking_reg =pd.read_sql(lots_parking,con=con1)
        parking_out =lots_parking_reg.iloc[0].values[0]
    return parking_out
def obtain_parking_distribution_strata (id_strate:int):
    eng = get_connection()
    with eng.connect() as con1:
        lots_query = f''' 
            SELECT 
                g_no_lot
            from association_strates 
            where id_strate ={id_strate}
        '''
        lots = pd.read_sql(lots_query,con=con1)
        lots_list = lots['g_no_lot'].unique().tolist()
        lots_parking = f'''
            SELECT
                g_no_lot,
                CEIL(n_places_min)::int as y_pred
            from inventaire_stationnement
            where g_no_lot in ('{"','".join(lots_list)}') and methode_estime=2
        '''
        lots_parking_reg =pd.read_sql(lots_parking,con=con1)
    return lots_parking_reg

def get_connection():
    env_data = de.load_dotenv()
    pg_host = os.environ.get('DB_HOST', 'localhost') #defaut localhost host.docker.internal
    pg_port = os.environ.get('DB_PORT', '5432') #defaut 5432
    pg_dbname = os.environ.get('DB_NAME', 'parking_regs_test')# specifique a l'application
    pg_username = os.environ.get('DB_USER', 'postgres') # defaut postgres
    pg_password = os.environ.get('DB_PASSWORD', 'admin') # specifique a l'application
    pg_string = 'postgresql://' + pg_username + ':'  + pg_password + '@'  + pg_host + ':'  + pg_port + '/'  + pg_dbname
    eng = create_engine(pg_string)
    return eng