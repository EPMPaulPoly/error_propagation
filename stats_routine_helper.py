import obtain_data as od
from scipy.stats import bootstrap
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
import sklearn.metrics as skm
def stats_routine():    
    # Load data
    strata = od.obtain_strata()
    val_data = od.obtain_data()
    pop_counts = od.obtain_population_sizes()

    results = []

    for _, strate in strata.iterrows():
        id_strate = strate['id_strate']

        # Filter validation data for this stratum
        val_data_s = val_data.loc[val_data['id_strate'] == id_strate].copy()
        val_data_s['error'] = val_data_s['y_obs'] - val_data_s['y_pred']
        errors_array = val_data_s['error'].values

        # Sample + population info
        n = len(val_data_s)
        N = pop_counts.loc[pop_counts['id_strate'] == id_strate, 'popu_strate'].iloc[0]
        scale_factor = N / n

        # Bootstrap mean error (more stable than sum)
        res = bootstrap((errors_array,), statistic=np.mean, n_resamples=2000, method='percentile')
        ci_l = res.confidence_interval.low
        ci_h = res.confidence_interval.high

        # Scale CI to total population
        scaled_ci_l = N * ci_l
        scaled_ci_h = N * ci_h

        # Compute sample and scaled totals
        total_pred_sample = val_data_s['y_pred'].sum()
        total_pred_scaled = scale_factor * total_pred_sample
        total_parking_pop = od.obtain_parking_estimate_strata(id_strate)

        # Store results
        results.append({
            'id_strate': id_strate,
            'N_pop': N,
            'n_samp': n,
            'total_spots_sample': total_pred_sample,
            'scaled_total_spots': total_pred_scaled,
            'sample_ci_l_mean_err': ci_l,
            'sample_ci_h_mean_err': ci_h,
            'scaled_ci_l_total_err': scaled_ci_l,
            'scaled_ci_h_total_err': scaled_ci_h,
            'total_parking_pred': total_parking_pop
        })

    df = pd.DataFrame(results)
    return df

def bootstrap_totals(val_data_check:pd.DataFrame,n_sample, n_lots, model_estimate_total:int,n_iterations:int=1000):
    # bootstrap 
    mean_residuals = np.mean(val_data_check['error'])
    scaled_mean_residual = mean_residuals * (n_lots/n_sample)
    bias_corrected_total = model_estimate_total + scaled_mean_residual
    centered_residuals = val_data_check['error'] - mean_residuals
    boot_totals = np.zeros(n_iterations)
    for i in range(n_iterations):
        boot_totals[i] = boot_total(centered_residuals, bias_corrected_total, n_lots, n_sample)
    # 95% confidence interval for the total population quantity
    ci_lower = np.percentile(boot_totals, 2.5)
    ci_upper = np.percentile(boot_totals, 97.5)
    # 95% confidence interval for the total population quantity
    print(f"Intervalle d'autoamorçage par percentile (95%) pour estimé du stationnement total: [{ci_lower:.2f}, {ci_upper:.2f}]")
    return {'ci_lower':ci_lower,
            'ci_upper':ci_upper,
            'boot_totals':boot_totals,
            'inv_biais_corrig':bias_corrected_total}

def boot_total(centered_residuals, bias_corrected_total, N_pop,n_sample):
    resampled_residuals = np.random.choice(centered_residuals, size=n_sample, replace=True)
    scaled_residual_sum = np.sum(resampled_residuals) * (N_pop / n_sample)
    adjusted_total = bias_corrected_total + scaled_residual_sum
    return adjusted_total

def bootstrap_error(val_data_check,n_sample,n_iterations:int=1000):
    boot_mean = np.zeros(n_iterations)
    boot_mae = np.zeros(n_iterations)
    residuals = val_data_check['error']
    mean_error = np.mean(residuals)
    mae = skm.mean_absolute_error(val_data_check['y_obs'],val_data_check['y_pred'])
    for i in range(n_iterations):
        data = boot_error(val_data_check, n_sample)
        boot_mean[i] = data['mean_error']
        boot_mae[i] = data['mae']
    # 95% prediction interval for the mean error
    me_pi_lower = np.percentile(boot_mean, 2.5)
    me_pi_upper = np.percentile(boot_mean, 97.5)
    print(f"Intervalle de prédiction par autoamorçage par percentile (95%) pour erreur moyenne: [{me_pi_lower:.2f}, {me_pi_upper:.2f}]")
    # 95% prediction interval for the mean error
    mae_pi_lower = np.percentile(boot_mae, 2.5)
    mae_pi_upper = np.percentile(boot_mae, 97.5)
    print(f"Intervalle de prédiction par autoamorçage par percentile (95%) pour erreur absolue moyenne: [{mae_pi_lower:.2f}, {mae_pi_upper:.2f}]")
    return {'me_pi_lower':me_pi_lower,
            'me_pi_upper':me_pi_upper,
            'me_bootstrap_dis':boot_mean,
            'me':mean_error,
            'mae_pi_lower':mae_pi_lower,
            'mae_pi_upper':mae_pi_upper,
            'mae_bootstrap_dis':boot_mae,
            'mae':mae,
            }
def boot_error(val_data_check:pd.DataFrame,n_sample):
    resampled_data = val_data_check.sample(n=n_sample,replace=True)

    average_error = np.mean(resampled_data['error'])
    mae = skm.mean_absolute_error(resampled_data['y_obs'],resampled_data['y_pred'])

    return {'mean_error':average_error,'mae':mae}

def t_stat_ci(confidence:float,samples:pd.DataFrame):
    mean = samples['error'].mean()
    se = stats.sem(samples['error'])

    n = len(samples)
    dof = n-1

def calcule_erreur(val_data:pd.DataFrame)->pd.DataFrame:
    val_data['error'] = val_data['y_obs'] - val_data['y_pred']
    return val_data

def calcule_erreur_carre(val_data:pd.DataFrame)->pd.DataFrame:
    val_data['error_squared'] = val_data['error']**2
    return val_data

def elimine_donnnees_aberrantes(val_data:pd.DataFrame,max_error:int,id_strate:int,strate_desc):
    donnees_utiles = val_data.loc[abs(val_data['error'])<max_error]
    donnees_aberrantes = val_data.loc[abs(val_data['error'])>=max_error]
    fig,ax = plt.subplots(ncols=2,nrows=1,figsize=[8.5,5],sharey=True)
    ylimabs = np.max(np.abs(val_data['error']))
    dirty_mean = np.mean(val_data['error'])
    clean_mean = np.mean(donnees_utiles['error'])
    val_data.plot(x='y_pred',y='error',xlabel='$y_{{pred}}$',ylabel='$e=y_{{obs}}-y_{{pred}}$',title='Sans filtre',ax=ax[0],ylim=[-ylimabs,ylimabs],kind='scatter')
    ax[0].axhline(dirty_mean,linestyle='--',color='red')
    ax[1].axhline(clean_mean,linestyle='--',color='red')
    donnees_utiles.plot(x='y_pred',y='error',xlabel='$y_{{pred}}$',ylabel='$e=y_{{obs}}-y_{{pred}}$',title='Avec filtre',ax=ax[1],ylim=[-ylimabs,ylimabs],kind='scatter')
    fig.suptitle(f'{strate_desc} - $e_{{max}}$ = {max_error}')
    fig.savefig(f'output/filtre_{id_strate}_e_{max_error}',dpi=300)
    return donnees_utiles,donnees_aberrantes

def calcule_erreur_moyenne_absolue(val_data:pd.DataFrame):
    return np.mean(val_data['error'].abs())

def calcule_erreur_bruitee(val_data:pd.DataFrame,bruit:float):
    val_data['y_obs_bruitee'] = val_data['y_obs'] + np.random.uniform(-bruit, bruit, len(val_data))
    val_data['erreur_bruitee'] = val_data['y_obs_bruitee'] - val_data['y_pred']
    return val_data

def single_strata(id_strate:int,bins:int=5,xlim:list[int]=None,max_error:int=None,n_it_range:list[int]=None,jitter:float=None):
    # Load data

    strata = od.obtain_strata() # strata titles
    val_data = od.obtain_data() # observed values for whole shebang
    pop_counts = od.obtain_population_sizes() # get population sizes from inputs table
    stat_total_categ =od.obtain_parking_estimate_strata(id_strate) #estime de la population total
    all_park = od.obtain_parking_distribution_strata(id_strate) # get all predictions in sample
    sample_input_values = od.obtain_sample_input_data(id_strate)
    #population_input_values = od.obtain_population_input_data(id_strate)
    n_iterations = 2000
    # Copie
    val_data_check = val_data.loc[val_data['id_strate']==id_strate].copy()
    val_data_check = calcule_erreur(val_data_check)
    val_data_check = calcule_erreur_carre(val_data_check)
    jitter_bool = False
    if jitter is not None:
        val_data_check = calcule_erreur_bruitee(val_data_check,jitter)
        jitter_bool = True
    if xlim is None:
        max_pred = int(val_data_check['y_pred'].max())
        xlim = [0,max_pred*1.2]
    
    ## -----------------------------------------------------------
    # taille échantillon et population et description
    ## -----------------------------------------------------------
    n_sample = len(val_data_check)
    
    n_lots = pop_counts.loc[pop_counts['id_strate']==id_strate,'popu_strate'].values[0]
    strat_desc = strata.loc[strata['id_strate']==id_strate,'desc_concat'].values[0]
    ## -----------------------------------------------------------
    # Éliminiation optionnel des propriétés aberrantes
    ## -----------------------------------------------------------
    if max_error is not None:
        val_data_check,don_aber = elimine_donnnees_aberrantes(val_data_check,max_error,id_strate,strat_desc)
        n_outliers = len(don_aber)
        print('n_outliers :',n_outliers)
    #-----------------------------------------
    # statistiques de base)
    # -------------------------------
    rmse = skm.root_mean_squared_error(val_data_check['y_obs'],val_data_check['y_pred'])
    mae = skm.mean_absolute_error(val_data_check['y_obs'],val_data_check['y_pred'])
    r2_score = skm.r2_score(val_data_check['y_obs'],val_data_check['y_pred'])
    #mape = skm.mean_absolute_percentage_error(val_data_check['y_obs'],val_data_check['y_pred'])
    ## -----------------------------------------------------------
    # statistique shapiro pour normalité et skewness pour asymétrie
    ## -----------------------------------------------------------
    if len(val_data_check) <= 5000:
        shap = stats.shapiro(val_data_check['error'])
    else:
        shap = SimpleNamespace(statistic=np.nan, pvalue=np.nan)
    skewness = stats.skew(val_data_check['error'])

    ## -----------------------------------------------------------
    # intervalle d'erreur bootstrap.
    ## -----------------------------------------------------------
    #bootstrap_return = bootstrap_totals(val_data_check, n_sample,n_lots,park_counts,n_iterations)
    bootstrap_return = bootstrap_error(val_data_check,n_sample,n_iterations)
    ## -----------------------------------------------------------
    # début graphiques
    ## -----------------------------------------------------------
    
    fig,ax = plt.subplots(nrows=2,ncols=4,figsize=[11,8.5],)
    # Titre figure
    sup_title_base =f'Catégorie: {strat_desc} - n= {n_sample} - N= {n_lots} - $\\sum y_{{pred}}$ = {stat_total_categ}'
    if max_error is not None:
        sup_title_base += f' - $e_{{max}}$={max_error}'
    if jitter_bool:
        sup_title_base += f' - Bruité'
    fig.suptitle(sup_title_base,fontsize=12)
    graphique_distro_erreur(val_data_check,ax[0,0],n_sample,bins)    
    graphique_residus(val_data_check,n_sample,ax[0,1],xlim,jitter_bool)
    graphique_residus_carre(val_data_check,n_sample,xlim,ax[0,2])
    graphique_predit_vs_obs(val_data_check,ax[0,3],jitter_bool)
    graphique_QQ(val_data_check,ax[1,0])
    graphique_distro_pred(val_data_check,all_park,ax[1,1])
    graphique_bootstrap_me(bootstrap_return,ax[1,2],n_iterations)
    graphique_bootstrap_mae(bootstrap_return,ax[1,3],n_iterations)
    fig.subplots_adjust(right=0.875,left=0.08,bottom=0.05,top=0.925,hspace=0.25,wspace=0.5)
    # Add separate axes for the text
    print_out_in_figure(fig,val_data_check,bootstrap_return,rmse,r2_score,shap,skewness,[0.9,0.4])
    base_name = f"output/int_erreur_{id_strate}"
    if max_error is not None:
        base_name += f"_e_{max_error}"
    if jitter_bool:
        base_name+="_b"
    base_name +=".png"
    fig.savefig(base_name, dpi=300)
    
    if n_it_range is not None:
        analyse_sensibilite_iterations_bootstrap(n_it_range,val_data_check,n_sample,n_lots,stat_total_categ,strat_desc)
    graphiques_analyse_residus(val_data_check,sample_input_values,strat_desc,jitter_bool,max_error)
    if id_strate==42:
        val_data_joined = val_data_check.copy().merge(sample_input_values.copy(),on='g_no_lot',how='left') 
        
        val_data_joined['categ'] = val_data_joined['list_unite'].apply(
            lambda x: (
                1 if isinstance(x, list) and 4 in x and len(x) == 1
                else 3 if isinstance(x, list) and 20 in x
                else 2 if isinstance(x, list) and 8 in x
                else None
            )
        )
        val_data_joined['categ_label'] = val_data_joined['categ'].map({
            1: 'Superficie seulement',
            2: 'Sièges',
            3: 'Trous de golf'
        })
        val_data_joined.boxplot(column='error', by='categ_label', grid=False)
        plt.suptitle('Erreur par unité de règlement')
        plt.title('')
        plt.xlabel('Catégorie')
        plt.ylabel('Erreur')
        plt.savefig("output/boxplot_error_by_category_42.png", dpi=300, bbox_inches="tight")
        #plt.show()
        


    
def graphiques_analyse_residus(val_data:pd.DataFrame,sample_input_values:pd.DataFrame,strat_desc:str,jitter_bool:bool=False,max_error:int=None):
    val_data_joined = val_data.copy().merge(sample_input_values.copy(),on='g_no_lot',how='left')
    fig,ax = plt.subplots(nrows=2,ncols=5,figsize=[11,8.5],sharey=True)
    graph_erreur_vs_aire_plancher(val_data_joined,ax[0,0],jitter_bool)
    graph_erreur_vs_date_constr(val_data_joined,ax[0,1],jitter_bool)
    graph_erreur_vs_distance_parlement(val_data_joined,ax[0,2],jitter_bool)
    graph_erreur_vs_unite_atypiques(val_data_joined,ax[0,3],jitter_bool)
    graph_erreur_vs_y_pred(val_data_joined,ax[0,4],jitter_bool)
    graph_erreur_vs_superf_lot(val_data_joined,ax[1,0],jitter_bool)
    graph_erreur_vs_n_log(val_data_joined,ax[1,1],jitter_bool)
    graph_erreur_vs_val_role(val_data_joined,ax[1,2],jitter_bool)
    #filtre pour données erronnées
    val_data_joined = val_data_joined.loc[val_data_joined['valeur_totale']>100].copy()
    graph_erreur_vs_val_m2(val_data_joined,ax[1,3],jitter_bool)
    graph_erreur_vs_y_pred_m2_par_val(val_data_joined,ax[1,4],jitter_bool)
    #graph_erreur_vs_y_pred_par_m2(val_data_joined,ax[1,5],jitter_bool)
    sup_title_base = f'Analyse Résidus - {strat_desc}'
    if jitter_bool is True:
        sup_title_base += ' - Bruités'
    if max_error is not None:
        sup_title_base += f' - $e_{{max}}$ = {max_error}'
    fig.suptitle(sup_title_base,fontsize=12)
    id_strate = val_data_joined['id_strate_x'].max()
    fig.subplots_adjust(right=0.95,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
    base_name = f'output/ana_res_{id_strate}'
    if max_error is not None:
        base_name += f"_e_{max_error}"
    if jitter_bool:
        base_name += f"_b"
    base_name+=".png"
    fig.savefig(base_name, dpi=300)

def graphique_distro_erreur(val_data:pd.DataFrame,ax:plt.axes,n_sample:int,bins:int):
    ## -----------------------------------------------------------
    # distribution erreurs
    ## -----------------------------------------------------------
    val_data['error'].hist(ax=ax, bins=bins,rwidth=0.8,grid=False,align='mid')
    ax.set_title(f'a) Distribution des erreurs - n= {n_sample}')
    ax.set_xlabel(f'$e=y_{{obs}}-y_{{pred}}$')
    ax.set_ylabel(f'Nombre de propriétés')

def graphique_QQ(val_data:pd.DataFrame,ax:plt.axes):
    ## -----------------------------------------------------------
    # diagramme q-q comparaison à une loi normale
    ## -----------------------------------------------------------
    stats.probplot(val_data['error'], dist="norm", plot=ax)
    ax.set_title(f'e) Q-Q distribution normale')
    ax.set_xlabel(f'Quantiles théoriques')
    ax.set_ylabel(f'Valeurs observées e')
    #plt.show()
def graphique_residus(val_data:pd.DataFrame,n_sample:int,ax:plt.axes,xlim:list[int],jitter:bool=False):
    ## -----------------------------------------------------------
    # predit vs residus : alternative tukey ou bland altman
    ## -----------------------------------------------------------
    if jitter:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(kind='scatter',x='y_pred',y='erreur_bruitee',xlabel='$y_{{pred}}$',ylabel='$e=y_{{obs}}-y_{{pred}}$',ax=ax,xlim=xlim,title=f'b) Prédit vs erreurs - n={n_sample}',ylim=ylim)
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(kind='scatter',x='y_pred',y='error',xlabel='$y_{{pred}}$',ylabel='$e=y_{{obs}}-y_{{pred}}$',ax=ax,xlim=xlim,title=f'b) Prédit vs erreurs - n={n_sample}',ylim=ylim)
    ax.axhline(0,color='red',linestyle='--')
def graphique_residus_carre(val_data:pd.DataFrame,n_sample:int,xlim:list[int],ax:plt.axes):
    ## -----------------------------------------------------------
    # prédit vs résidus au carré voir si on peut faire une prédiction sur l'entier positif
    ## -----------------------------------------------------------
    val_data.plot(kind='scatter',x='y_pred',y='error_squared',xlabel='$y_{{pred}}$',ylabel='$e^2=(y_{{obs}}-y_{{pred}})^2$',ax=ax,xlim=xlim,title=f'c) Prédit vs erreurs au carré - n={n_sample}')

def analyse_sensibilite_iterations_bootstrap(n_it_range:list[int],val_data:pd.DataFrame,n_sample:int,n_lots:int,park_counts:int,strat_desc:str):
    iteration_range = np.linspace(n_it_range[0],n_it_range[1],250)
    ci_l_it_check = []
    ci_h_it_check = []
    for n_its in iteration_range:
        bootstrap_it = bootstrap(val_data,n_sample,n_lots,park_counts,int(n_its))
        ci_l_it_check.append(bootstrap_it['ci_lower'])
        ci_h_it_check.append(bootstrap_it['ci_upper'])
    fig2,ax2 = plt.subplots(figsize=[5,5])
    ax2.plot(iteration_range,ci_l_it_check,color='blue')
    ax2.plot(iteration_range,ci_h_it_check,color='red')
    ax2.set_title(f"Convergence de l'autoamorçage - Strate: {strat_desc} - n= {n_sample} - N= {n_lots} - Stat = {park_counts}")
    ax2.set_xlabel("Nombre d'itérations")
    ax2.set_ylabel("Valeur des intervalles de confiance")


def graphique_bootstrap_me(bootstrap_return,ax:plt.axes,n_iterations):
    ## -----------------------------------------------------------
    # graphiques de l'intervalle d'erreur bootstrap
    ## -----------------------------------------------------------
    # capture histogram artists so legend refers to the correct handle
    n_vals, bins_vals, patches = ax.hist(bootstrap_return['me_bootstrap_dis'], bins=10, rwidth=0.8, align='mid')
    hist_patch = patches[0] if len(patches) > 0 else None
    line_inv = ax.axvline(x=0, color='violet', linestyle='-')
    line_bias = ax.axvline(x=bootstrap_return['me'], color='cyan', linestyle='--')
    line_ci_l = ax.axvline(x=bootstrap_return['me_pi_lower'], color='lime', linestyle='-')
    line_ci_h = ax.axvline(x=bootstrap_return['me_pi_upper'], color='red', linestyle='-')
    ax.set_title(f'g) Autoamorçage - {n_iterations} iter \nErreur Moyenne')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['AA', f'Zero', f'ME = {bootstrap_return['me']:.2f}', f'$ME_{{bas}}$ = {bootstrap_return['me_pi_lower']:.2f}', f'$ME_{{haut}}$={bootstrap_return['me_pi_upper']:.2f}']
    ax.set_xlabel('ME')
    ax.set_ylabel('$N_{{iter}}$')
    xlim = [-np.max(np.abs(bootstrap_return['me_bootstrap_dis']))*1.25,np.max(np.abs(bootstrap_return['me_bootstrap_dis']))*1.25]
    ax.set_xlim(xlim[0],xlim[1])
    ax.legend(handles=handles, labels=labels[:len(handles)])

def graphique_bootstrap_mae(bootstrap_return,ax:plt.axes,n_iterations):
    ## -----------------------------------------------------------
    # graphiques de l'intervalle d'erreur bootstrap
    ## -----------------------------------------------------------
    # capture histogram artists so legend refers to the correct handle
    n_vals, bins_vals, patches = ax.hist(bootstrap_return['mae_bootstrap_dis'], bins=10, rwidth=0.8, align='mid')
    hist_patch = patches[0] if len(patches) > 0 else None
    line_inv = ax.axvline(x=0, color='violet', linestyle='-')
    line_bias = ax.axvline(x=bootstrap_return['mae'], color='cyan', linestyle='--')
    line_ci_l = ax.axvline(x=bootstrap_return['mae_pi_lower'], color='lime', linestyle='-')
    line_ci_h = ax.axvline(x=bootstrap_return['mae_pi_upper'], color='red', linestyle='-')
    ax.set_title(f'h) Autoamorçage - {n_iterations} iter \n Erreur absolue moyenne')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['AA', f'Zero', f'MAE = {bootstrap_return['mae']:.2f}', f'$MAE_{{bas}}$ = {bootstrap_return['mae_pi_lower']:.2f}', f'$MAE_{{haut}}$={bootstrap_return['mae_pi_upper']:.2f}']
    ax.set_xlabel('MAE')
    ax.set_ylabel('$N_{{iter}}$')
    ax.legend(handles=handles, labels=labels[:len(handles)])

def graphique_distro_pred(val_data,all_park,ax):
    ## -----------------------------------------------------------
    ## comparaison de la distribution des valeurs prédites 
    ## -----------------------------------------------------------
    # plot y_pred distributions with same bins on the remaining subplot (ax[1,2])
    s1 = val_data['y_pred']
    # try to extract a y_pred series from all_park (works if it's a Series or a DataFrame)
    if isinstance(all_park, pd.DataFrame):
        s2 = all_park['y_pred'] if 'y_pred' in all_park.columns else all_park.iloc[:, 0]
    elif isinstance(all_park, pd.Series):
        s2 = all_park
    else:
        s2 = pd.Series(all_park).squeeze()

    combined_min = min(s1.min(), s2.min())
    combined_max = max(np.percentile(s1, 90), np.percentile(s2, 90))
    # guard against zero range
    if combined_min == combined_max:
        combined_min -= 0.5
        combined_max += 0.5
    bin_edges = np.linspace(combined_min, combined_max, 10 + 1)

    ax.hist(s1, bins=bin_edges, alpha=0.6, label='Validation', rwidth=0.8,density=True,align='mid')
    ax.hist(s2, bins=bin_edges, alpha=0.4, label='Toutes Prédictions', rwidth=0.8,density=True,align='mid')
    ax.set_title('f) Distribution des prédictions')
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Fréquence')
    ax.legend()

def graphique_predit_vs_obs(val_data,ax,jitter_bool:bool=False):
    ## -----------------------------------------------------------
    # prédiv vs obs. devrait être une ligne droite.
    ## -----------------------------------------------------------
    if jitter_bool:
        val_data.plot(kind='scatter',x='y_pred',y='y_obs_bruitee',ax=ax,xlabel='$y_{{pred}}$',ylabel='$y_{{obs}}+\\epsilon$',title='d) Observé vs prédit')
    else:
        val_data.plot(kind='scatter',x='y_pred',y='y_obs',ax=ax,xlabel='$y_{{pred}}$',ylabel='$y_{{obs}}$',title='d) Observé vs prédit')
    ax.axline((0, 0), (val_data['y_obs'].max(), val_data['y_obs'].max()), linewidth=4, color='r')

def print_out_in_figure(fig:plt.figure,val_data,bootstrap_return,rmse,r2_score,shap,skew,loc):
    fig.text(loc[0],loc[1],f"""Shapiro $e$ = {shap.statistic:.2f}\n Skew $e$ = {skew:.2f}\n $\\bar{{y}}_{{obs}}$= {np.mean(val_data['y_obs']):.2f}\n $\\bar{{y}}_{{pred}}$ = {np.mean(val_data['y_pred']):.2f}\nME = {bootstrap_return['me']:.2f}  \n RMSE = {rmse:.2f}\n MAE = {bootstrap_return['mae']:.2f} """,fontsize=10)


def graph_erreur_vs_aire_plancher(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='sup_planch_tot',y='erreur_bruitee',xlabel='Aire plancher',ylabel='$y_{{obs}}-y_{{pred}} + \\epsilon$',kind='scatter',ax=ax,ylim=ylim,title='a)')
    else:    
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='sup_planch_tot',y='error',xlabel='Aire plancher',ylabel='$y_{{obs}}-y_{{pred}}$',kind='scatter',ax=ax,ylim=ylim,title='a)')
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_date_constr(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='premiere_constr',y='erreur_bruitee',xlabel='Date de construction',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,title='b)')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='premiere_constr',y='error',xlabel='Date de construction',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,title='b)')
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_distance_parlement(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='dist_to_parliament',y='erreur_bruitee',xlabel='Distance Parlement',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,title='c)')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='dist_to_parliament',y='error',xlabel='Distance Parlement',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,title='c)')
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_superf_lot(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='superf_lot',y='erreur_bruitee',xlabel='Superficie Lot',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,title="f)")
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='superf_lot',y='error',xlabel='Superficie Lot',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,title='f)')
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_n_log(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='n_logements_tot',y='erreur_bruitee',xlabel='Nombre de logements',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim)
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='n_logements_tot',y='error',xlabel='Nombre de logements',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim)
    n_tot = len(val_data)
    n_non_null = int(val_data.loc[~val_data['n_logements_tot'].isna()].count().values[0])
    ax.set_title(f'g) $n_{{tot}}$ = {n_tot} $n_{{non-na}}$ = {n_non_null}')
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_val_role(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='valeur_totale',y='erreur_bruitee',xlabel='Valeur au rôle',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,title='h)')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='valeur_totale',y='error',xlabel='Valeur au rôle',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,title='h)')
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_y_pred(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='y_pred',y='erreur_bruitee',xlabel='Prédiction',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,title='e)')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='y_pred',y='error',xlabel='Prédiction',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,title='e)')
    ax.axhline(0,color='red',linestyle="--")


def graph_erreur_vs_val_m2(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    val_data_add = val_data.copy()
    val_data_add['val_m2'] = val_data_add['valeur_totale']/val_data_add['superf_lot']
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee'])),np.max(np.abs(val_data['erreur_bruitee']))]
        val_data_add.plot(x='val_m2',y='erreur_bruitee',xlabel=r'$\frac{\mathrm{\$}}{m^{2}_{\mathrm{lot}}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,title='i)')
    else:
        ylim = [-np.max(np.abs(val_data['error'])),np.max(np.abs(val_data['error']))]
        val_data_add.plot(x='val_m2',y='error',xlabel=r'$\frac{\mathrm{\$}}{m^{2}_{\mathrm{lot}}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,title='i)')
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_y_pred_m2_par_val(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    val_data_add = val_data.copy()
    val_data_add['predict'] = val_data_add['y_pred']/val_data_add['valeur_totale']*val_data_add['superf_lot']
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee'])),np.max(np.abs(val_data['erreur_bruitee']))]
        val_data_add.plot(x='predict',y='erreur_bruitee',xlabel=r'$\frac{y_{pred}\times m^{2}_{\mathrm{lot}}}{\mathrm{\$}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,title='j)')
    else:
        ylim = [-np.max(np.abs(val_data['error'])),np.max(np.abs(val_data['error']))]
        val_data_add.plot(x='predict',y='error',xlabel=r'$\frac{y_{pred}\times m^{2}_{\mathrm{lot}}}{\mathrm{\$}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,title='j)')
    ax.axhline(0,color='red',linestyle="--")
def graph_erreur_vs_unite_atypiques(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee'])),np.max(np.abs(val_data['erreur_bruitee']))]
        # Make the boxplot
        val_data.boxplot(
            column='erreur_bruitee',
            by='atypical_units',
            ax=ax,
            ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',
            xlabel='Unités atypiques?',
            grid=False
        )
    else:
        ylim = [-np.max(np.abs(val_data['error'])),np.max(np.abs(val_data['error']))]
        # Make the boxplot
        val_data.boxplot(
            column='error',
            by='atypical_units',
            ax=ax,
            ylabel='$y_{{obs}}-y_{{pred}}$',
            xlabel='Unités atypiques?',
            grid=False
        )
    
    ax.set_ylim(ylim[0],ylim[1])
    
    # Count points per group
    counts = val_data['atypical_units'].value_counts().sort_index()

    # Replace x-tick labels with counts
    ax.set_xticklabels([f"{cat}\n(n={counts[cat]})" for cat in counts.index])
    ax.set_title("d)")
    ax.axhline(0,color='red',linestyle="--")

def graph_erreur_vs_y_pred_par_m2(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False):
    val_data_add = val_data.copy()
    val_data_add['predict'] = val_data_add['y_pred']/val_data_add['superf_lot']
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee'])),np.max(np.abs(val_data['erreur_bruitee']))]
        val_data_add.plot(x='predict',y='erreur_bruitee',xlabel=r'$\frac{y_{pred}}{m^{2}_{\mathrm{lot}}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim)
    else:
        ylim = [-np.max(np.abs(val_data['error'])),np.max(np.abs(val_data['error']))]
        val_data_add.plot(x='predict',y='error',xlabel=r'$\frac{y_{pred}}{m^{2}_{\mathrm{lot}}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim)
    ax.axhline(0,color='red',linestyle="--")