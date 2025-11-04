import obtain_data as od
from scipy.stats import bootstrap
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
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
    residuals = val_data_check['error']
    mean_error = np.mean(residuals)
    for i in range(n_iterations):
        boot_mean[i] = boot_average_error(residuals, n_sample)
    # 95% confidence interval for the total population quantity
    ci_lower = np.percentile(boot_mean, 2.5)
    ci_upper = np.percentile(boot_mean, 97.5)
    # 95% confidence interval for the total population quantity
    print(f"Intervalle de prédiction par autoamorçage par percentile (95%) pour estimé erreur: [{ci_lower:.2f}, {ci_upper:.2f}]")
    return {'ci_lower':ci_lower,
            'ci_upper':ci_upper,
            'error_bootstrap_dis':boot_mean,
            'erreur_moyenne':mean_error}
def boot_average_error(residuals,n_sample):
    resampled_residuals = np.random.choice(residuals, size=n_sample, replace=True)
    average_error = np.mean(resampled_residuals)
    return average_error

def t_stat_ci(confidence:float,samples:pd.DataFrame):
    mean = samples['error'].mean()
    se = stats.sem(samples['error'])

    n = len(samples)
    dof = n-1

def calcule_erreur(val_data:pd.DataFrame)->pd.DataFrame:
    val_data['error'] = val_data['y_obs'] - val_data['y_pred']
    return val_data

def calcule_erreur_carre(val_data:pd.DataFrame)->pd.DataFrame:
    val_data['error_squared'] = val_data['error_squared']**2
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
    n_iterations = 2000
    # Copie
    val_data_check = val_data.loc[val_data['id_strate']==id_strate].copy()
    val_data_check = calcule_erreur(val_data_check)
    val_data_check = calcule_erreur_carre(val_data_check)
    ## -----------------------------------------------------------
    # Éliminiation optionnel des propriétés aberrantes
    ## -----------------------------------------------------------
    if max_error is not None:
        val_data_check,don_aber = elimine_donnnees_aberrantes(val_data_check,max_error)
    ## -----------------------------------------------------------
    # taille échantillon et population et description
    ## -----------------------------------------------------------
    n_sample = len(val_data_check)
    n_outliers = len(don_aber)
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
    
    graphique_distro_erreur(val_data_check,ax,n_sample,bins)
    graphique_QQ(val_data_check,shap,skewness,ax)
    graphique_residus(val_data_check,n_sample,ax,xlim)
    graphique_residus_carre(val_data_check,n_sample,xlim,ax)
    graphique_bootstrap(bootstrap_return,stat_total_categ,ax,n_iterations)
    graphique_distro_pred(val_data_check,all_park,ax)
    graphique_predit_vs_obs(val_data_check,ax)
    print(shap.statistic)
    if n_it_range is not None:
        analyse_sensibilite_iterations_bootstrap(n_it_range,val_data_check,n_sample,n_lots,stat_total_categ,strat_desc)
    ax[1,3].axis('off')    

def graphique_distro_erreur(val_data:pd.DataFrame,ax:plt.axes,n_sample:int,bins:int):
    ## -----------------------------------------------------------
    # distribution erreurs
    ## -----------------------------------------------------------
    val_data['error'].hist(ax=ax[0,0], bins=bins,rwidth=0.8,grid=False,align='mid')
    ax[0,0].set_title(f'Distribution des erreurs - n= {n_sample}')
    ax[0,0].set_xlabel(f'obs-pred')
    ax[0,0].set_ylabel(f'Nombre de propriété')

def graphique_QQ(val_data:pd.DataFrame,shap:float,skewness:float,ax:plt.axes):
    ## -----------------------------------------------------------
    # diagramme q-q comparaison à une loi normale
    ## -----------------------------------------------------------
    stats.probplot(val_data['error'], dist="norm", plot=ax[1,0])
    ax[1,0].set_title(f'Q-Q - SW= {shap.statistic:.2f} - Skew= {skewness:.2f}')
    ax[1,0].set_xlabel(f'Quantiles théoriques')
    ax[1,0].set_ylabel(f'Valeurs observées')
    #plt.show()
def graphique_residus(val_data:pd.DataFrame,n_sample:int,ax:plt.axes,xlim:list[int]):
    ## -----------------------------------------------------------
    # predit vs residus : alternative tukey ou bland altman
    ## -----------------------------------------------------------
    val_data.plot(kind='scatter',x='y_pred',y='error',xlabel='Stationnement prédit',ylabel='obs-pred',ax=ax[0,1],xlim=xlim,title=f'Prédit vs erreurs - n={n_sample}')

def graphique_residus_carre(val_data:pd.DataFrame,n_sample:int,xlim:list[int],ax:plt.axes):
    ## -----------------------------------------------------------
    # prédit vs résidus au carré voir si on peut faire une prédiction sur l'entier positif
    ## -----------------------------------------------------------
    val_data.plot(kind='scatter',x='y_pred',y='error_squared',xlabel='Stationnement prédit',ylabel='$(obs-pred)^2$',ax=ax[0,2],xlim=xlim,title=f'Prédit vs erreurs au carré - n={n_sample}')

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


def graphique_bootstrap(bootstrap_return,park_counts,ax:plt.axes,n_iterations):
    ## -----------------------------------------------------------
    # graphiques de l'intervalle d'erreur bootstrap
    ## -----------------------------------------------------------
    # capture histogram artists so legend refers to the correct handle
    n_vals, bins_vals, patches = ax[1,1].hist(bootstrap_return['error_bootstrap_dis'], bins=10, rwidth=0.8, align='mid')
    hist_patch = patches[0] if len(patches) > 0 else None
    line_inv = ax[1,1].axvline(x=0, color='violet', linestyle='-')
    line_bias = ax[1,1].axvline(x=bootstrap_return['erreur_moyenne'], color='cyan', linestyle='--')
    line_ci_l = ax[1,1].axvline(x=bootstrap_return['ci_lower'], color='lime', linestyle='-')
    line_ci_h = ax[1,1].axvline(x=bootstrap_return['ci_upper'], color='red', linestyle='-')
    ax[1,1].set_title(f'Autoamorçage - {n_iterations} iter - Stationnement total')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['Autoamorçage', f'Erreur=0', f'Erreur Moyenne = {bootstrap_return['erreur_moyenne']:.2f}', f'IC bas = {bootstrap_return['ci_lower']:.2f}', f'IC haut={bootstrap_return['ci_upper']:.2f}']
    ax[1,1].legend(handles=handles, labels=labels[:len(handles)])

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