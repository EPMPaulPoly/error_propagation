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
    print(f"Intervalle de prédiction par autoamorçage par percentile (95%) pour erreur moyenne absolue: [{mae_pi_lower:.2f}, {mae_pi_upper:.2f}]")
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
    mae = skm.mean_absolute_error(val_data_check['y_obs'],val_data_check['y_pred'])

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

def elimine_donnnees_aberrantes(val_data:pd.DataFrame,max_error:int):
    donnees_utiles = val_data.loc[abs(val_data['error'])<max_error]
    donnees_aberrantes = val_data.loc[abs(val_data['error'])>=max_error]
    return donnees_utiles,donnees_aberrantes

def calcule_erreur_moyenne_absolue(val_data:pd.DataFrame):
    return np.mean(val_data['error'].abs())


def single_strata(id_strate:int,xlim:list[int]=[0,10],bins:int=5,max_error:int=None,n_it_range:list[int]=None):
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

    #-----------------------------------------
    # statistiques de base)
    # -------------------------------
    rmse = skm.root_mean_squared_error(val_data_check['y_obs'],val_data_check['y_pred'])
    mae = skm.mean_absolute_error(val_data_check['y_obs'],val_data_check['y_pred'])
    r2_score = skm.r2_score(val_data_check['y_obs'],val_data_check['y_pred'])
    #mape = skm.mean_absolute_percentage_error(val_data_check['y_obs'],val_data_check['y_pred'])
    ## -----------------------------------------------------------
    # Éliminiation optionnel des propriétés aberrantes
    ## -----------------------------------------------------------
    if max_error is not None:
        val_data_check,don_aber = elimine_donnnees_aberrantes(val_data_check,max_error)
        n_outliers = len(don_aber)
    ## -----------------------------------------------------------
    # taille échantillon et population et description
    ## -----------------------------------------------------------
    n_sample = len(val_data_check)
    
    n_lots = pop_counts.loc[pop_counts['id_strate']==id_strate,'popu_strate'].values[0]
    strat_desc = strata.loc[strata['id_strate']==id_strate,'desc_concat'].values[0]
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
    
    fig,ax = plt.subplots(nrows=2,ncols=4,figsize=[10,10])
    # Titre figure
    fig.suptitle(f'Strate: {strat_desc} - n= {n_sample} - N= {n_lots} - Stat = {stat_total_categ}')
    
    graphique_distro_erreur(val_data_check,ax[0,0],n_sample,bins)
    graphique_QQ(val_data_check,shap,skewness,ax[1,0])
    graphique_residus(val_data_check,n_sample,ax[0,1],xlim)
    graphique_residus_carre(val_data_check,n_sample,xlim,ax[0,2])
    graphique_bootstrap_me(bootstrap_return,ax[1,1],n_iterations)
    graphique_distro_pred(val_data_check,all_park,ax)
    graphique_predit_vs_obs(val_data_check,ax)
    print_out_in_figure(fig,val_data_check,bootstrap_return,rmse,mae,r2_score)
    print(shap.statistic)
    ax[1,3].axis('off')
    
    if n_it_range is not None:
        analyse_sensibilite_iterations_bootstrap(n_it_range,val_data_check,n_sample,n_lots,stat_total_categ,strat_desc)
    graphiques_analyse_residus(val_data_check,sample_input_values,strat_desc)
    

def graphique_distro_erreur(val_data:pd.DataFrame,ax:plt.axes,n_sample:int,bins:int):
    ## -----------------------------------------------------------
    # distribution erreurs
    ## -----------------------------------------------------------
    val_data['error'].hist(ax=ax, bins=bins,rwidth=0.8,grid=False,align='mid')
    ax.set_title(f'Distribution des erreurs - n= {n_sample}')
    ax.set_xlabel(f'obs-pred')
    ax.set_ylabel(f'Nombre de propriété')

def graphique_QQ(val_data:pd.DataFrame,shap:float,skewness:float,ax:plt.axes):
    ## -----------------------------------------------------------
    # diagramme q-q comparaison à une loi normale
    ## -----------------------------------------------------------
    stats.probplot(val_data['error'], dist="norm", plot=ax)
    ax.set_title(f'Q-Q - SW= {shap.statistic:.2f} - Skew= {skewness:.2f}')
    ax.set_xlabel(f'Quantiles théoriques')
    ax.set_ylabel(f'Valeurs observées')
    #plt.show()
def graphique_residus(val_data:pd.DataFrame,n_sample:int,ax:plt.axes,xlim:list[int]):
    ## -----------------------------------------------------------
    # predit vs residus : alternative tukey ou bland altman
    ## -----------------------------------------------------------
    val_data.plot(kind='scatter',x='y_pred',y='error',xlabel='Stationnement prédit',ylabel='obs-pred',ax=ax,xlim=xlim,title=f'Prédit vs erreurs - n={n_sample}')

def graphique_residus_carre(val_data:pd.DataFrame,n_sample:int,xlim:list[int],ax:plt.axes):
    ## -----------------------------------------------------------
    # prédit vs résidus au carré voir si on peut faire une prédiction sur l'entier positif
    ## -----------------------------------------------------------
    val_data.plot(kind='scatter',x='y_pred',y='error_squared',xlabel='Stationnement prédit',ylabel='$(obs-pred)^2$',ax=ax,xlim=xlim,title=f'Prédit vs erreurs au carré - n={n_sample}')

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
    ax.set_title(f'Autoamorçage - {n_iterations} iter - Erreur Moyenne')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['Autoamorçage', f'Erreur=0', f'Erreur Moyenne = {bootstrap_return['me']:.2f}', f'IP bas = {bootstrap_return['me_pi_lower']:.2f}', f'IP haut={bootstrap_return['me_pi_upper']:.2f}']
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
    ax.set_title(f'Autoamorçage - {n_iterations} iter - Erreur absolue moyenne')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['Autoamorçage', f'Erreur=0', f'Erreur Moyenne = {bootstrap_return['mae']:.2f}', f'IP bas = {bootstrap_return['mae_pi_lower']:.2f}', f'IP haut={bootstrap_return['mae_pi_upper']:.2f}']
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

    ax[1,2].hist(s1, bins=bin_edges, alpha=0.6, label='Validation', rwidth=0.8,density=True,align='mid')
    ax[1,2].hist(s2, bins=bin_edges, alpha=0.4, label='Toutes Prédictions', rwidth=0.8,density=True,align='mid')
    ax[1,2].set_title('Distribution des prédictions')
    ax[1,2].set_xlabel('Prédiction')
    ax[1,2].set_ylabel('Fréquence')
    ax[1,2].legend()

def graphique_predit_vs_obs(val_data,ax):
    ## -----------------------------------------------------------
    # prédiv vs obs. devrait être une ligne droite.
    ## -----------------------------------------------------------
    val_data.plot(kind='scatter',x='y_pred',y='y_obs',ax=ax[0,3])
    ax[0,3].axline((0, 0), (val_data['y_obs'].max(), val_data['y_obs'].max()), linewidth=4, color='r')

def print_out_in_figure(fig:plt.figure,val_data,bootstrap_return,rmse,mae,r2_score):
    fig.text(0.8,0.15,f"""Places/lot obs moy = {np.mean(val_data['y_obs']):.2f}\n Places/lot pred moy = {np.mean(val_data['y_pred']):.2f}\nME = {bootstrap_return['erreur_moyenne']:.2f} places \n RMSE = {rmse:.2f}\n MAE = {mae:.2f}\n  $R^2$ = {r2_score:.2f}""")

def graphiques_analyse_residus(val_data:pd.DataFrame,sample_input_values:pd.DataFrame,strat_desc:str):
    val_data_joined = val_data.copy().merge(sample_input_values.copy(),on='g_no_lot',how='left')
    fig,ax = plt.subplots(nrows=2,ncols=5,figsize=[10,5])
    graph_erreur_vs_aire_plancher(val_data_joined,ax[0,0])
    graph_erreur_vs_date_constr(val_data_joined,ax[0,1])
    graph_erreur_vs_distance_parlement(val_data_joined,ax[0,2])
    graph_erreur_vs_unite_atypiques(val_data_joined,ax[0,3])
    graph_erreur_vs_y_pred(val_data_joined,ax[0,4])
    graph_erreur_vs_superf_lot(val_data_joined,ax[1,0])
    graph_erreur_vs_n_log(val_data_joined,ax[1,1])
    graph_erreur_vs_val_role(val_data_joined,ax[1,2])
    graph_erreur_vs_val_m2(val_data_joined,ax[1,3])
    graph_erreur_vs_y_pred_m2_par_val(val_data_joined,ax[1,4])
    fig.suptitle(f'Analyse Résidus - {strat_desc}')

def graph_erreur_vs_aire_plancher(val_data:pd.DataFrame,ax:plt.axes):
    val_data.plot(x='sup_planch_tot',y='error',xlabel='Aire plancher',ylabel='$y_{{obs}}-y_{{pred}}$',kind='scatter',ax=ax)

def graph_erreur_vs_date_constr(val_data:pd.DataFrame,ax:plt.axes):
    val_data.plot(x='premiere_constr',y='error',xlabel='Date de construction',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)

def graph_erreur_vs_distance_parlement(val_data:pd.DataFrame,ax:plt.axes):
    val_data.plot(x='dist_to_parliament',y='error',xlabel='Distance Parlement',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)

def graph_erreur_vs_superf_lot(val_data:pd.DataFrame,ax:plt.axes):
    val_data.plot(x='superf_lot',y='error',xlabel='Superficie Lot',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)

def graph_erreur_vs_n_log(val_data:pd.DataFrame,ax:plt.axes):
    val_data.plot(x='n_logements_tot',y='error',xlabel='Nombre de logements',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)
    n_tot = len(val_data)
    n_non_null = int(val_data.loc[~val_data['n_logements_tot'].isna()].count().values[0])
    ax.set_title(f'$n_{{tot}}$ = {n_tot} $n_{{non-na}}$ = {n_non_null}')

def graph_erreur_vs_val_role(val_data:pd.DataFrame,ax:plt.axes):
    val_data.plot(x='valeur_totale',y='error',xlabel='Valeur au rôle',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)

def graph_erreur_vs_y_pred(val_data:pd.DataFrame,ax:plt.axes):
    val_data.plot(x='y_pred',y='error',xlabel='Prédiction',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)


def graph_erreur_vs_val_m2(val_data:pd.DataFrame,ax:plt.axes):
    val_data_add = val_data.copy()
    val_data_add['val_m2'] = val_data_add['valeur_totale']/val_data_add['superf_lot']
    val_data_add.plot(x='val_m2',y='error',xlabel=r'$\frac{\mathrm{\$}}{m^{2}_{\mathrm{lot}}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)

def graph_erreur_vs_y_pred_m2_par_val(val_data:pd.DataFrame,ax:plt.axes):
    val_data_add = val_data.copy()
    val_data_add['predict'] = val_data_add['y_pred']/val_data_add['valeur_totale']*val_data_add['superf_lot']
    val_data_add.plot(x='predict',y='error',xlabel=r'$\frac{y_{pred}\times m^{2}_{\mathrm{lot}}}{\mathrm{\$}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax)
def graph_erreur_vs_unite_atypiques(val_data:pd.DataFrame,ax:plt.axes):
    # Make the boxplot
    val_data.boxplot(
        column='error',
        by='atypical_units',
        ax=ax,
        ylabel='$y_{{obs}}-y_{{pred}}$',
        xlabel='Unités atypiques?'
    )
    
    # Count points per group
    counts = val_data['atypical_units'].value_counts().sort_index()

    # Replace x-tick labels with counts
    ax.set_xticklabels([f"{cat}\n(n={counts[cat]})" for cat in counts.index])
    ax.set_title("")

