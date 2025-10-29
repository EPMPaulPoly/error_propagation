import obtain_data as od
from scipy.stats import bootstrap
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def bootstrap(val_data_check:pd.DataFrame,n_sample, n_lots, model_estimate_total:int,n_iterations:int=1000):
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
    print(f"95% Confidence Interval for Total Population Quantity: [{ci_lower:.2f}, {ci_upper:.2f}]")
    return {'ci_lower':ci_lower,
            'ci_upper':ci_upper,
            'boot_totals':boot_totals,
            'inv_biais_corrig':bias_corrected_total}

def boot_total(centered_residuals, bias_corrected_total, N_pop,n_sample):
    resampled_residuals = np.random.choice(centered_residuals, size=n_sample, replace=True)
    scaled_residual_sum = np.sum(resampled_residuals) * (N_pop / n_sample)
    adjusted_total = bias_corrected_total + scaled_residual_sum
    return adjusted_total

def t_stat_ci(confidence:float,samples:pd.DataFrame):
    mean = samples['error'].mean()
    se = stats.sem(samples['error'])

    n = len(samples)
    dof = n-1

    

def single_strata(id_strate:int,xlim:list[int]=[0,10],bins:int=5,max_error:int=None,n_it_range:list[int]=None):
    # Load data
    strata = od.obtain_strata()
    val_data = od.obtain_data()
    pop_counts = od.obtain_population_sizes()
    park_counts =od.obtain_parking_estimate_strata(id_strate)
    
    # Copie
    val_data_check = val_data.loc[val_data['id_strate']==id_strate].copy()
    
    # calcul des résidus
    val_data_check['error'] = val_data_check['y_obs'] - val_data_check['y_pred']
    val_data_check['error_squared'] = val_data_check['error']**2
    if max_error is not None:
        val_data_check = val_data_check.loc[abs(val_data_check['error'])<max_error]
    n_sample = len(val_data_check)
    n_lots = pop_counts.loc[pop_counts['id_strate']==id_strate,'popu_strate'].values[0]
    strat_desc = strata.loc[strata['id_strate']==id_strate,'desc_concat'].values[0]
    # statistique shapiro pour normalité
    shap = stats.shapiro(val_data_check['error'])
    skewness = stats.skew(val_data_check['error'])
    # graphiques
    fig,ax = plt.subplots(nrows=2,ncols=3,figsize=[10,10])
    # Titre figure
    fig.suptitle(f'Strate: {strat_desc} - n= {n_sample} - N= {n_lots} - Stat = {park_counts}')
    # distribution erreurs
    val_data_check['error'].hist(ax=ax[0,0], bins=bins,rwidth=0.8,grid=False,align='mid')
    ax[0,0].set_title(f'Distribution des erreurs - n= {n_sample}')
    ax[0,0].set_xlabel(f'Obs-pred')
    ax[0,0].set_ylabel(f'Nombre de propriété')
    # diagramme q-q
    stats.probplot(val_data_check['error'], dist="norm", plot=ax[1,0])
    ax[1,0].set_title(f'Q-Q - SW= {shap.statistic:.2f} - Skew= {skewness:.2f}')
    ax[1,0].set_xlabel(f'Quantiles théoriques')
    ax[1,0].set_ylabel(f'Valeurs observées')
    # predit vs residus
    val_data_check.plot(kind='scatter',x='y_pred',y='error',xlabel='Stationnement prédit',ylabel='Obs-pred',ax=ax[0,1],xlim=xlim,title=f'Prédit vs erreurs - n={n_sample}')

    # prédit vs résidus au carré
    val_data_check.plot(kind='scatter',x='y_pred',y='error_squared',xlabel='Stationnement prédit',ylabel='$(Obs-pred)^2$',ax=ax[0,2],xlim=xlim,title=f'Prédit vs erreurs au carré - n={n_sample}')
    print(shap.statistic)
    
    n_iterations = 2000
    if n_it_range is not None:
        iteration_range = 0

    bootstrap_return = bootstrap(val_data_check, n_sample,n_lots,park_counts,n_iterations)
    #ci_normal = stats.t.interval(0.95,)
    # capture histogram artists so legend refers to the correct handle
    n_vals, bins_vals, patches = ax[1,1].hist(bootstrap_return['boot_totals'], bins=10, rwidth=0.8, align='mid')
    hist_patch = patches[0] if len(patches) > 0 else None
    line_inv = ax[1,1].axvline(x=park_counts, color='violet', linestyle='-')
    line_bias = ax[1,1].axvline(x=bootstrap_return['inv_biais_corrig'], color='cyan', linestyle='--')
    line_ci_l = ax[1,1].axvline(x=bootstrap_return['ci_lower'], color='lime', linestyle='-')
    line_ci_h = ax[1,1].axvline(x=bootstrap_return['ci_upper'], color='red', linestyle='-')
    ax[1,1].set_title(f'Bootstrap - {n_iterations} iter')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['Bootstrap', f'Inventaire modèle = {park_counts}', f'Inventaire biais-corrigé = {bootstrap_return['inv_biais_corrig']:.2f}', f'IC bas = {bootstrap_return['ci_lower']:.2f}', f'IC haut={bootstrap_return['ci_upper']:.2f}']
    ax[1,1].legend(handles=handles, labels=labels[:len(handles)])
    #plt.show()





