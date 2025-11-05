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

def obtain_sample_input_data(id_strate:int):
    eng = get_connection()
    query =f'''
        with lot_ass as (
            SELECT
                *
            FROM
                assignation_strates ass
            WHERE
                id_strate = %(id)s
        ), lot_values AS(
            SELECT
                g_no_lot,
                nb_entrees,
                cubf_presents,
                sup_planch_tot,
                n_logements_tot,
                premiere_constr,
                valeur_totale,
                cubf_principal
            FROM
                inputs_validation
            where 
                g_no_lot in (select g_no_lot from lot_ass)
        ), distance_to_parliament AS(
            SELECT
                g_no_lot,
				ST_Transform(ST_Centroid(geometry),4326) as lot_centroid,
				ST_Point(-71.21417,46.80861,4326) as parliament,
                ST_distance(ST_Transform(ST_Centroid(geometry),32198),ST_Transform(ST_Point(-71.21417,46.80861,4326),32198)) as dist_to_parliament,
				g_va_suprf as superf_lot
            FROM
                cadastre
            WHERE g_no_lot IN(select g_no_lot from lot_ass)
        ),reg_inventory AS (
			SELECT
				g_no_lot,
				CEIL(n_places_min) as y_pred,
				regexp_split_to_array(id_reg_stat, '[,/]')::int[] AS list_id_reg,
				regexp_split_to_array(id_er, '[,/]')::int[] AS list_id_er
			FROM 
				inventaire_stationnement
			where g_no_lot in (select g_no_lot from lot_ass) and methode_estime=2
		), units AS (
			SELECT 
				ri.g_no_lot,
				ri.list_id_reg,
				ri.list_id_er,
				ARRAY_AGG(DISTINCT rse.unite) as list_unite,
				ARRAY[1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] && ARRAY_AGG(DISTINCT rse.unite) as atypical_units 
			FROM 
				reg_inventory ri
			left join reg_stationnement_empile rse on rse.id_reg_stat = any(ri.list_id_reg)
			group by ri.list_id_reg,ri.list_id_er,ri.g_no_lot
		)
        SELECT
			lv.*,
			la.id_strate,
			dtp.dist_to_parliament,
			dtp.lot_centroid,
			dtp.parliament,
            dtp.superf_lot,
			u.list_id_reg,
			u.list_id_er,
			u.list_unite,
			u.atypical_units
        from lot_ass la
        LEFT JOIN lot_values lv on lv.g_no_lot = la.g_no_lot
        left join distance_to_parliament dtp on dtp.g_no_lot=la.g_no_lot 
		left join units u on u.g_no_lot = la.g_no_lot
    '''
    with eng.connect() as con1:
        input_data = pd.read_sql_query(query,con=con1,params={"id":id_strate})
    return input_data

def obtain_population_input_data(id_strate:int):
    eng = get_connection()
    query =f'''
        with lot_ass as (
            SELECT
                *
            FROM
                association_strates ass
            WHERE
                id_strate = %(id)s
        ), lot_values AS(
            SELECT
                g_no_lot,
                nb_entrees,
                cubf_presents,
                sup_planch_tot,
                n_logements_tot,
                premiere_constr,
                valeur_totale,
                cubf_principal
            FROM
                inputs_validation
            where 
                g_no_lot in (select g_no_lot from lot_ass)
        ), distance_to_parliament AS(
            SELECT
                g_no_lot,
				ST_Transform(ST_Centroid(geometry),4326) as lot_centroid,
				ST_Point(-71.21417,46.80861,4326) as parliament,
                ST_distance(ST_Transform(ST_Centroid(geometry),32198),ST_Transform(ST_Point(-71.21417,46.80861,4326),32198)) as dist_to_parliament,
				g_va_suprf as superf_lot
            FROM
                cadastre
            WHERE g_no_lot IN(select g_no_lot from lot_ass)
        ),reg_inventory AS (
			SELECT
				g_no_lot,
				CEIL(n_places_min) as y_pred,
				regexp_split_to_array(id_reg_stat, '[,/]')::int[] AS list_id_reg,
				regexp_split_to_array(id_er, '[,/]')::int[] AS list_id_er
			FROM 
				inventaire_stationnement
			where g_no_lot in (select g_no_lot from lot_ass) and methode_estime=2
		), units AS (
			SELECT 
				ri.g_no_lot,
				ri.list_id_reg,
				ri.list_id_er,
				ARRAY_AGG(DISTINCT rse.unite) as list_unite,
				ARRAY[1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21] && ARRAY_AGG(DISTINCT rse.unite) as atypical_units 
			FROM 
				reg_inventory ri
			left join reg_stationnement_empile rse on rse.id_reg_stat = any(ri.list_id_reg)
			group by ri.list_id_reg,ri.list_id_er,ri.g_no_lot
		)
        SELECT
			lv.*,
			la.id_strate,
			dtp.dist_to_parliament,
			dtp.lot_centroid,
			dtp.parliament,
            dtp.superf_lot,
			u.list_id_reg,
			u.list_id_er,
			u.list_unite,
			u.atypical_units
        from lot_ass la
        LEFT JOIN lot_values lv on lv.g_no_lot = la.g_no_lot
        left join distance_to_parliament dtp on dtp.g_no_lot=la.g_no_lot 
		left join units u on u.g_no_lot = la.g_no_lot
    '''
    with eng.connect() as con1:
        input_data = pd.read_sql_query(query,con=con1,params={"id":id_strate})
    return input_data

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