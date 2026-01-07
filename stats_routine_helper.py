from matplotlib.ticker import FormatStrFormatter
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
    ylimabs = np.max(np.abs(val_data['error']))*1.25
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

def single_strata(id_strate:int,bins:int=5,xlim:list[int]=None,max_error:int=None,n_it_range:list[int]=None,jitter:float=None,perc_error:float=None,spot_error:int=None,interval_plots=False,error_plots=False,unit_plots=False):
    # Load data

   # strata = od.obtain_strata() # strata titles
    #val_data = od.obtain_data() # observed values for whole shebang
    #pop_counts = od.obtain_population_sizes() # get population sizes from inputs table
    #stat_total_categ =od.obtain_parking_estimate_strata(id_strate) #estime de la population total
    #all_park = od.obtain_parking_distribution_strata(id_strate) # get all predictions in sample
    #sample_input_values = od.obtain_sample_input_data(id_strate)
    [strata,val_data,pop_counts,stat_total_categ,all_park,sample_input_values] = od.obtain_overall_data(id_strate)
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
    
    
    ## -----------------------------------------------------------
    # taille échantillon et population et description
    ## -----------------------------------------------------------
    n_sample = len(val_data_check)
    
    n_lots = pop_counts.loc[pop_counts['id_strate']==id_strate,'popu_strate'].values[0]
    strat_desc = strata.loc[strata['id_strate']==id_strate,'desc_concat'].values[0]
    ## -----------------------------------------------------------
    # Éliminiation optionnel des propriétés aberrantes
    ## -----------------------------------------------------------
    n_outliers= None
    if max_error is not None:
        val_data_check,don_aber = elimine_donnnees_aberrantes(val_data_check,max_error,id_strate,strat_desc)
        n_outliers = len(don_aber)
        print('n_outliers :',n_outliers)
    if xlim is None:
        max_pred = int(val_data_check['y_pred'].max())
        xlim = [0,max_pred*1.05]
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
    if interval_plots:
        graphique_distr_intervalles(
            val_data_check,
            n_sample,
            n_lots,
            stat_total_categ,
            n_outliers,bins,
            xlim,jitter_bool,
            bootstrap_return,
            all_park,
            n_iterations,
            id_strate,
            rmse,
            r2_score,
            shap,
            skewness,
            strat_desc,
            max_error,
            spot_error,
            perc_error
            )
    if error_plots:
        graphiques_analyse_residus(val_data_check,sample_input_values,strat_desc,jitter_bool,max_error,n_sample,n_outliers,spot_error,perc_error)
    if unit_plots:
        graphique_unites(val_data_check,sample_input_values,max_error,id_strate)
    if n_it_range is not None:
        analyse_sensibilite_iterations_bootstrap(n_it_range,val_data_check,n_sample,n_lots,stat_total_categ,strat_desc)
    plt.close('all')
        
def graphique_unites(val_data_check,sample_input_values,max_error,id_strate):
    val_data_joined = val_data_check.copy().merge(sample_input_values.copy(),on='g_no_lot',how='left') 
        
    val_data_joined['categ'] = val_data_joined['list_unite'].apply(
        lambda x: (
            1 if isinstance(x, list) and 4 in x and len(x) == 1
            else 3 if isinstance(x, list) and 20 in x
            else 2 if isinstance(x, list) and 8 in x
            else 4 if isinstance(x,list) and 14 in x
            else 5 if isinstance(x,list) and 13 in x
            else 6 if isinstance(x,list) and 10 in x
            else 7 if isinstance(x,list) and 17 in x
            else 8 if isinstance(x,list) and 1 in x
            else -1
        )
    )

    val_data_joined['categ_label'] = val_data_joined['categ'].map({
        1: 'Superficie seulement',
        2: 'Sièges',
        3: 'Trous de golf',
        4: 'Étudiant',
        5: 'Employé',
        6: 'Salle',
        7: 'Voie de service',
        8: 'Chambre',
        -1: 'Autre'
    })
    ax=val_data_joined.boxplot(column='error', by='categ_label', grid=False)
    ax.axhline(0,color='red',linestyle='--')
    plt.suptitle("Erreur selon l'unité utilisée dans le règlement")
    plt.title('')
    plt.xlabel('Unités utilisées dans le règlement')
    plt.ylabel(r'$e = y_{\mathrm{obs}} - y_{\mathrm{pred}}$')
    base_name_2 = f'boxplot_error_by_category_{id_strate}'
    if max_error is not None:
        base_name +=f"_e_{max_error}"
    plt.savefig(f"output/{base_name_2}.png", dpi=300, bbox_inches="tight")
    #plt.show()
def graphique_distr_intervalles(val_data_check,n_sample,n_lots,stat_total_categ,n_outliers,bins,xlim,jitter_bool,bootstrap_return,all_park,n_iterations,id_strate,rmse,r2_score,shap,skewness,strat_desc,max_error,spot_error,perc_error):
    ## -----------------------------------------------------------
    # début graphiques
    ## -----------------------------------------------------------
    
    # Titre figure
    sup_title_base =f'Catégorie: {strat_desc} - n= {n_sample} - N= {n_lots} - $\\sum y_{{pred}}$ = {stat_total_categ}'
    if max_error is not None:
        sup_title_base += f' - $e_{{max}}$={max_error} - $n_{{filtre}}$ = {n_sample-n_outliers}'
    if jitter_bool:
        sup_title_base += f' - Bruité'
    fig1,ax1 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],)
    fig1.suptitle(sup_title_base+' - Survol',fontsize=12)
    graphique_predit_vs_obs(val_data_check,ax1[0],jitter_bool,spot_error=spot_error,perc_error=perc_error)
     
    graphique_distro_pred(val_data_check,all_park,ax1[1])
    fig1.subplots_adjust(right=0.875,left=0.08,bottom=0.075,top=0.925,hspace=0.3,wspace=0.2)
    print_out_base_stats(fig1,val_data_check,r2_score,[0.11,0.75])

    fig2,ax2 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],)
    fig2.suptitle(sup_title_base+' - Échelle',fontsize=12)
    graphique_residus(val_data_check,n_sample,ax2[0],xlim,jitter_bool,spot_error_bnd=spot_error,pct_error_bnd=perc_error)
    graphique_residus_carre(val_data_check,n_sample,xlim,ax2[1],spot_bnd_error=spot_error,perc_bnd_error=perc_error)
    fig2.subplots_adjust(right=0.875,left=0.08,bottom=0.075,top=0.925,hspace=0.3,wspace=0.2)

    fig3,ax3 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],)
    fig3.suptitle(sup_title_base+' - Distribution',fontsize=12)
    graphique_QQ(val_data_check,ax3[0])
    
    graphique_distro_erreur(val_data_check,ax3[1],n_sample,bins,spot_error_bnd=spot_error)   
    fig3.subplots_adjust(right=0.95,left=0.15,bottom=0.075,top=0.925,hspace=0.3,wspace=0.2)
    print_out_distr_info(fig3,shap,skewness,[0.01,0.35])

    fig4,ax4 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],)
    fig4.suptitle(sup_title_base+' - Auto-Amorçage',fontsize=12)
    graphique_bootstrap_me(bootstrap_return,ax4[0],n_iterations)
    graphique_bootstrap_mae(bootstrap_return,ax4[1],n_iterations)
    fig4.subplots_adjust(right=0.875,left=0.08,bottom=0.075,top=0.875,hspace=0.3,wspace=0.2)
    print_out_errors_info(fig4,bootstrap_return,rmse,[0.9,0.4])
    # Add separate axes for the text
    #print_out_in_figure(fig4,val_data_check,bootstrap_return,rmse,r2_score,shap,skewness,[0.9,0.4])
    base_name = f"output/int_erreur_{id_strate}"
    if max_error is not None:
        base_name += f"_e_{max_error}"
    if jitter_bool:
        base_name+="_b"
    if spot_error is not None:
        base_name+=f"_se_{spot_error:.0f}"
    if perc_error is not None:
        base_name+=f"_pe_{(100*perc_error):.0f}"
    fig1.savefig(base_name+'_a.png', dpi=300)
    fig2.savefig(base_name+'_b.png', dpi=300)
    fig3.savefig(base_name+'_c.png', dpi=300)
    fig4.savefig(base_name+'_d.png', dpi=300)
    
def graphiques_analyse_residus(val_data:pd.DataFrame,sample_input_values:pd.DataFrame,strat_desc:str,jitter_bool:bool=False,max_error:int=None,n_sample:int=None,n_outliers:int=None,spot_error:int=None,perc_error:float=None):
    val_data_joined = val_data.copy().merge(sample_input_values.copy(),on='g_no_lot',how='left')
    # Formation titre
    sup_title_base = f'Analyse Résidus - {strat_desc}'
    if n_sample is not None:
        sup_title_base += f" - n= {n_sample}"
    if n_outliers is not None:
        sup_title_base += f" - $n_{{filtre}}$= {n_sample-n_outliers}"
    if jitter_bool is True:
        sup_title_base += ' - Bruités'
    if max_error is not None:
        sup_title_base += f' - $e_{{max}}$ = {max_error}'
    # Règlements
    fig1,ax1 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],sharey=True)
    graph_erreur_vs_aire_plancher(val_data_joined,ax1[0],jitter_bool,spot_error)
    graph_erreur_vs_n_log(val_data_joined,ax1[1],fig1,jitter_bool,spot_error)
    fig1.suptitle(sup_title_base+' - Variables typiques',fontsize=12)
    # Espace temps
    fig2,ax2 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],sharey=True)
    graph_erreur_vs_date_constr(val_data_joined,ax2[0],jitter_bool,spot_error)
    graph_erreur_vs_distance_parlement(val_data_joined,ax2[1],jitter_bool,spot_error)       
    fig2.suptitle(sup_title_base+' - Espace Temps',fontsize=12)
    # Unités Échelles
    fig3,ax3 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],sharey=True)
    graph_erreur_vs_unite_atypiques(val_data_joined,ax3[0],jitter_bool,spot_error)
    graph_erreur_vs_y_pred(val_data_joined,ax3[1],jitter_bool,spot_error,perc_error)
    fig3.suptitle(sup_title_base+' - Unités Échelle',fontsize=12)

    # Cash et lot
    fig4,ax4 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],sharey=True)
    graph_erreur_vs_superf_lot(val_data_joined,ax4[0],jitter_bool,spot_error)
    graph_erreur_vs_val_role(val_data_joined,ax4[1],jitter_bool,spot_error)
    fig4.suptitle(sup_title_base+' - Valeur et taille de lot',fontsize=12)
    #filtre pour données erronnées
    
    val_data_joined = val_data_joined.loc[val_data_joined['valeur_totale']>100].copy()
    # fig 5: Intensité
    fig5,ax5 = plt.subplots(nrows=1,ncols=2,figsize=[13,8.5],sharey=True)
    graph_erreur_vs_val_m2(val_data_joined,ax5[0],jitter_bool,spot_error)
    graph_erreur_vs_y_pred_m2_par_val(val_data_joined,ax5[1],jitter_bool,spot_error)
    fig5.suptitle(sup_title_base+' - Densité foncière et intensité prédiction',fontsize=12)
    #graph_erreur_vs_y_pred_par_m2(val_data_joined,ax[1,5],jitter_bool)
    
    
    id_strate = val_data_joined['id_strate_x'].max()
    if spot_error is not None or perc_error is not None:
        fig1.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig2.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig3.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig4.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig5.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
    else:
        fig1.subplots_adjust(right=0.95,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig2.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig3.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig4.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)
        fig5.subplots_adjust(right=0.925,left=0.08,bottom=0.1,top=0.925,hspace=0.25,wspace=0.1)

    # sauvegarde
    base_name = f'output/ana_res_{id_strate}'
    if max_error is not None:
        base_name += f"_e_{max_error}"
    if jitter_bool:
        base_name += f"_b"
    if spot_error is not None:
        base_name+=f"_se_{spot_error:.0f}"
    if perc_error is not None:
        base_name+=f"_pe_{(100*perc_error):.0f}"
    fig1.savefig(base_name+'_a.png', dpi=300)
    fig2.savefig(base_name+'_b.png', dpi=300)
    fig3.savefig(base_name+'_c.png', dpi=300)
    fig4.savefig(base_name+'_d.png', dpi=300)
    fig5.savefig(base_name+'_e.png', dpi=300)

def graphique_distro_erreur(val_data:pd.DataFrame,ax:plt.axes,n_sample:int,bins:int,spot_error_bnd:int=None):
    ## -----------------------------------------------------------
    # distribution erreurs
    ## -----------------------------------------------------------
    errors = val_data['error']

    ax.hist(errors, bins=bins, rwidth=0.8, align='mid',
            color='steelblue', edgecolor='black',label='Distribution')
    ax.grid(False)

    # symmetric x ticks around 0
    max_abs = np.nanmax(np.abs(errors))
    lim = np.ceil(max_abs * 1.05)
    ax.set_xlim(-lim, lim)

    # set symmetric ticks
    xticks = np.linspace(-lim, lim, 9)  # 9 evenly spaced ticks
    ax.set_xticks(xticks)

    ax.set_title(f'a) Distribution des erreurs')
    ax.set_xlabel(r'$e = y_{\mathrm{obs}} - y_{\mathrm{pred}}$  [pl.]')
    ax.set_ylabel('Nombre de propriétés [-]')
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='both', labelsize=9, width=0.8, direction='out')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # optional vertical line at zero for clarity
    ax.axvline(0, color='red', linewidth=1, linestyle='--',label='$e=0$ pl.')
    if spot_error_bnd is not None:
        ax.axvline(spot_error_bnd, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e =+{spot_error_bnd:.0f}$ pl.')
        ax.axvline(-spot_error_bnd, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e =-{spot_error_bnd:.0f}$ pl.')
    ax.legend(loc='center left', bbox_to_anchor=(0.05, 0.9), frameon=True, fontsize=10)

def graphique_QQ(val_data:pd.DataFrame,ax:plt.axes):
    ## -----------------------------------------------------------
    # diagramme q-q comparaison à une loi normale
    ## -----------------------------------------------------------
    stats.probplot(val_data['error'], dist="norm", plot=ax)
    ax.set_title(f'a) Q-Q distribution normale')
    ax.set_xlabel(f'Quantiles théoriques loi normale')
    ax.set_ylabel(f'Quantiles observés e')
    ax.legend(['Distribution constatée','Distribution normale théorique'])
    #plt.show()

def graphique_residus(val_data:pd.DataFrame,n_sample:int,ax:plt.axes,xlim:list[int],jitter:bool=False,spot_error_bnd:int=None,pct_error_bnd:float=None):
    ## -----------------------------------------------------------
    # predit vs residus : alternative tukey ou bland altman
    ## -----------------------------------------------------------
    if jitter:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(kind='scatter',x='y_pred',y='erreur_bruitee',xlabel='$y_{{pred}}$ [pl.]',ylabel='$e=y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,xlim=xlim,title=f'a) Prédit vs erreurs',ylim=ylim,label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(kind='scatter',x='y_pred',y='error',xlabel='$y_{{pred}}$ [pl.]',ylabel='$e=y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,xlim=xlim,title=f'a) Prédit vs erreurs',ylim=ylim,label='Données')
    ax.axhline(0,color='red',linestyle='--',label='$e = 0$ pl.')
    if spot_error_bnd is not None:
        ax.axhline(spot_error_bnd, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e =+{spot_error_bnd:.0f}$ pl.')
        ax.axhline(-spot_error_bnd, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e =-{spot_error_bnd:.0f}$ pl.')
    if pct_error_bnd is not None:
        x_max = val_data['y_pred'].max()
        ax.axline((0,0),(x_max,x_max*pct_error_bnd),color='#ff7f0e',linestyle='--',label=f'$e= +{pct_error_bnd*100:.0f}$ $\%$')
        ax.axline((0,0),(x_max,-x_max*pct_error_bnd),color='#ff7f0e',linestyle='-.',label=f'$e= -{pct_error_bnd*100:.0f}$ $\%$')
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.99), frameon=True, fontsize=10)

def graphique_residus_carre(val_data:pd.DataFrame,n_sample:int,xlim:list[int],ax:plt.axes,spot_bnd_error:int=None,perc_bnd_error:float=None):
    ## -----------------------------------------------------------
    # prédit vs résidus au carré voir si on peut faire une prédiction sur l'entier positif
    ## -----------------------------------------------------------
    val_data.plot(kind='scatter',x='y_pred',y='error_squared',xlabel='$y_{{pred}}$ [pl.]',ylabel='$e^2=(y_{{obs}}-y_{{pred}})^2$  [pl.$^2$]',ax=ax,xlim=xlim,title=f'b) Prédit vs erreurs au carré - n={n_sample}',label='Données')  
    # Define line x-range
    xs = np.linspace(xlim[0], xlim[1], 200)

    # Add fixed ±error bound curve (spot error)
    if spot_bnd_error is not None:
        error_to_plot = (xs + spot_bnd_error - xs)**2  # (constant absolute error)
        ax.plot(xs, error_to_plot, '--', color='#2ca02c',label=f'$e={spot_bnd_error:.0f}$ pl.')

    # Add percentage-based bound curve (proportional error)
    if perc_bnd_error is not None:
        perc_error_to_plot = (xs * perc_bnd_error)**2
        ax.plot(xs, perc_error_to_plot, '--', color='#ff7f0e',label=f'$e={100*perc_bnd_error:.0f}$ \%')
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.99), frameon=True, fontsize=10)

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
    n_vals, bins_vals, patches = ax.hist(bootstrap_return['me_bootstrap_dis'], bins=10, rwidth=0.8, align='mid',edgecolor='black')
    ymax = np.max(n_vals)
    hist_patch = patches[0] if len(patches) > 0 else None
    line_inv = ax.axvline(x=0, color='violet', linestyle='-')
    line_bias = ax.axvline(x=bootstrap_return['me'], color='cyan', linestyle='--')
    line_ci_l = ax.axvline(x=bootstrap_return['me_pi_lower'], color='lime', linestyle='-')
    line_ci_h = ax.axvline(x=bootstrap_return['me_pi_upper'], color='red', linestyle='-')
    ax.set_title(f'a) Autoamorçage - {n_iterations} iter \nErreur Moyenne')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['AA', f'Zero', f'ME = {bootstrap_return['me']:.2f}', f'$ME_{{bas}}$ = {bootstrap_return['me_pi_lower']:.2f}', f'$ME_{{haut}}$={bootstrap_return['me_pi_upper']:.2f}']
    ax.set_xlabel('ME')
    ax.set_ylabel('$N_{{iter}}$')
    xlim = [-np.max(np.abs(bootstrap_return['me_bootstrap_dis']))*1.25,np.max(np.abs(bootstrap_return['me_bootstrap_dis']))*1.25]
    ax.set_xlim(xlim[0],xlim[1])
    ax.set_ylim(0,ymax*1.6)
    ax.legend(handles=handles, labels=labels[:len(handles)])

def graphique_bootstrap_mae(bootstrap_return,ax:plt.axes,n_iterations):
    ## -----------------------------------------------------------
    # graphiques de l'intervalle d'erreur bootstrap
    ## -----------------------------------------------------------
    # capture histogram artists so legend refers to the correct handle
    n_vals, bins_vals, patches = ax.hist(bootstrap_return['mae_bootstrap_dis'], bins=10, rwidth=0.8, align='mid',edgecolor='black')
    ymax = np.max(n_vals)
    hist_patch = patches[0] if len(patches) > 0 else None
    line_inv = ax.axvline(x=0, color='violet', linestyle='-')
    line_bias = ax.axvline(x=bootstrap_return['mae'], color='cyan', linestyle='--')
    line_ci_l = ax.axvline(x=bootstrap_return['mae_pi_lower'], color='lime', linestyle='-')
    line_ci_h = ax.axvline(x=bootstrap_return['mae_pi_upper'], color='red', linestyle='-')
    ax.set_title(f'b) Autoamorçage - {n_iterations} iter \n Erreur absolue moyenne')
    # build handles list skipping None values (in case hist produced no patches)
    handles = [h for h in [hist_patch, line_inv, line_bias, line_ci_l, line_ci_h] if h is not None]
    labels = ['AA', f'Zero', f'MAE = {bootstrap_return['mae']:.2f}', f'$MAE_{{bas}}$ = {bootstrap_return['mae_pi_lower']:.2f}', f'$MAE_{{haut}}$={bootstrap_return['mae_pi_upper']:.2f}']
    ax.set_xlabel('MAE')
    ax.set_ylabel('$N_{{iter}}$')
    ax.set_ylim(0,ymax*1.6)
    ax.legend(handles=handles, labels=labels[:len(handles)])

def graphique_distro_pred(val_data, all_park, ax, max_unique_for_bar=12, bar_offset_frac=0.4, bar_width_frac=0.4):
    """
    Compare predicted value distributions side by side.
    
    Automatically switches to bar plot if predictions are discrete (few unique values).
    
    Parameters
    ----------
    val_data : pd.DataFrame
        Validation data containing 'y_pred'.
    all_park : pd.DataFrame or pd.Series
        All predicted values.
    ax : plt.Axes
        Axis to plot on.
    max_unique_for_bar : int
        Max number of unique values to treat as discrete (switch to bar plot).
    bar_offset_frac : float
        Fraction of bin width to offset the bars for side-by-side plotting.
    bar_width_frac : float
        Fraction of bin width used for each bar width.
    """
    # --- Extract and clean numeric series ---
    s1 = pd.to_numeric(val_data['y_pred'], errors='coerce').dropna()
    if isinstance(all_park, pd.DataFrame):
        s2 = pd.to_numeric(all_park['y_pred'], errors='coerce').dropna() if 'y_pred' in all_park.columns else pd.to_numeric(all_park.iloc[:, 0], errors='coerce').dropna()
    elif isinstance(all_park, pd.Series):
        s2 = pd.to_numeric(all_park, errors='coerce').dropna()
    else:
        s2 = pd.Series(all_park).astype(float).dropna()

    # --- Check for discrete/few unique values ---
    unique_vals = np.union1d(s1.unique(), s2.unique())
    discrete = len(unique_vals) <= max_unique_for_bar

    if discrete:
        # --- Bar plot for discrete predictions ---
        counts1 = s1.value_counts().reindex(unique_vals, fill_value=0)/s1.sum()
        counts2 = s2.value_counts().reindex(unique_vals, fill_value=0)/s2.sum()
        combined_min = min(s1.min(), s2.min())
        combined_max = max(np.percentile(s1, 99), np.percentile(s2, 99))
        width = bar_width_frac
        offset = bar_offset_frac
        ax.set_xticks(unique_vals)
        ax.set_xlim([0.5, combined_max+0.5])
        ax.bar(unique_vals - offset/2, counts1, width=width, alpha=0.7, label='Validation',
               color='steelblue', edgecolor='black')
        ax.bar(unique_vals + offset/2, counts2, width=width, alpha=0.7, label='Toutes Prédictions',
               color='orange', edgecolor='black')
        y_max = np.max([counts1.max(), counts2.max()])

    else:
        # --- Continuous predictions: side-by-side histogram ---
        combined_min = min(s1.min(), s2.min())
        combined_max = max(np.percentile(s1, 90), np.percentile(s2, 90))
        if combined_min == combined_max:
            combined_min -= 0.5
            combined_max += 0.5

        n_bins = 10
        bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)
        bin_width = bin_edges[1] - bin_edges[0]

        hist1, _ = np.histogram(s1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(s2, bins=bin_edges, density=True)
        y_max=np.max([np.max(hist1),np.max(hist2)])
        offset = bin_width * bar_offset_frac

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.bar(bin_centers - offset/2, hist1, width=bin_width*bar_width_frac, alpha=0.6,
               label='Validation', color='steelblue', edgecolor='black', align='center')
        ax.bar(bin_centers + offset/2, hist2, width=bin_width*bar_width_frac, alpha=0.6,
               label='Toutes Prédictions', color='orange', edgecolor='black', align='center')

        # x-axis scaled to actual positive range
        ax.set_xlim(1-bin_width, combined_max)
        #xticks = np.linspace(combined_min, combined_max, 9)
        ax.set_xticks(bin_centers)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # --- Labels, legend, rotation ---
    ax.set_ylim(0,y_max*1.25)
    ax.set_title('b) Distribution des prédictions')
    ax.set_xlabel('Prédiction [pl.]')
    ax.set_ylabel('Fréquence')
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend()

def graphique_predit_vs_obs(val_data,ax,jitter_bool:bool=False,spot_error:int=None,perc_error:float=None):
    ## -----------------------------------------------------------
    # prédiv vs obs. devrait être une ligne droite.
    ## -----------------------------------------------------------
    
    if jitter_bool:
        lim = max([val_data['y_pred'].max(),val_data['y_obs_bruitee'].max()])*1.05
        val_data.plot(kind='scatter',x='y_pred',y='y_obs_bruitee',ax=ax,xlabel='$y_{{pred}}$',ylabel='$y_{{obs}}+\\epsilon$',title='a) Observé vs prédit',zorder=3,label='Données',xlim=[-1,lim],ylim=[-1,lim])
    else:
        lim = max([val_data['y_pred'].max(),val_data['y_obs'].max()])*1.05
        val_data.plot(kind='scatter',x='y_pred',y='y_obs',ax=ax,xlabel='$y_{{pred}}$ [pl.]',ylabel='$y_{{obs}}$ [pl.]',title='a) Observé vs prédit',zorder=3,label='Données',xlim=[-1,lim],ylim=[-1,lim])
    # Plot range
    x_min = -1
    x_max = lim * 1.05
    #ax.set_xlim(x_min, x_max)
    #x.set_ylim(x_min, x_max)

    # Reference line (perfect prediction)
    #ax.plot([x_min, x_max], [x_min, x_max], 'r-', linewidth=1.5, )

    ax.axline((0, 0), (lim, lim), linewidth=1, color='r',linestyle='--',label='$e=0$ pl.')
    # Constant error bands (±5 and ±10)
    if spot_error is not None:
        ax.plot([x_min, x_max], [x_min + spot_error, x_max + spot_error], linestyle='--', color='#2ca02c', linewidth=1, label=f'$e=+{spot_error:.0f}$ pl.')
        ax.plot([x_min, x_max], [x_min - spot_error, x_max - spot_error], linestyle='-.', color='#2ca02c', linewidth=1, label=f'$e=-{spot_error:.0f}$ pl.')

    if perc_error is not None:
        ax.plot([x_min, x_max], [x_min * (1 + perc_error), x_max * (1 + perc_error)], linestyle='--', color='#ff7f0e', linewidth=1, label=f'$e=+{int(perc_error*100):.0f}$ \%')
        ax.plot([x_min, x_max], [x_min * (1 - perc_error), x_max * (1 - perc_error)], linestyle='-.', color='#ff7f0e', linewidth=1, label=f'$e=-{int(perc_error*100):.0f}$ \%')

    # Legend on the right outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, .99), frameon=True, fontsize=10)
    
    ax.set_ylim(-1, lim)
    ax.set_xlim(-1, lim)
    # Optional grid and square aspect
    #ax.grid(True, linestyle=':', alpha=0.5)
    #ax.set_aspect('equal', 'box')

def print_out_in_figure(fig:plt.figure,val_data,bootstrap_return,rmse,r2_score,shap,skew,loc):
    fig.text(loc[0],loc[1],f"""Shapiro $e$ = {shap.statistic:.2f}\n Skew $e$ = {skew:.2f}\n $\\bar{{y}}_{{obs}}$= {np.mean(val_data['y_obs']):.2f}\n $\\bar{{y}}_{{pred}}$ = {np.mean(val_data['y_pred']):.2f}\nME = {bootstrap_return['me']:.2f}  \n RMSE = {rmse:.2f}\n MAE = {bootstrap_return['mae']:.2f} """,fontsize=10)

def print_out_base_stats(fig:plt.figure,val_data,r2_score,loc):
    fig.text(
        loc[0],
        loc[1],
        f"""$ \\bar{{y}}_{{obs}}$= {np.mean(val_data['y_obs']):.2f}\n $\\bar{{y}}_{{pred}}$ = {np.mean(val_data['y_pred']):.2f}\n $R^2$= {r2_score:.2f}""",
        fontsize=10,
        ha='left',                 # horizontal alignment (optional)
        va='top',                  # vertical alignment (optional)
        bbox=dict(
            facecolor='white',     # background colour inside the box
            edgecolor='gray',     # colour of the border line
            linewidth=1,           # thickness of the border
            alpha=0.3,
            boxstyle='round,pad=0.3'  # rounded corners + inner padding
        ))

def print_out_distr_info(fig:plt.figure,shap,skew,loc):
    fig.text(loc[0],loc[1],f"""Shapiro $e$ = {shap.statistic:.2f}\n Skew $e$ = {skew:.2f}\n""",fontsize=10)

def print_out_errors_info(fig:plt.figure,bootstrap_return,rmse,loc):
    fig.text(loc[0],loc[1],f"""ME = {bootstrap_return['me']:.2f}  \n RMSE = {rmse:.2f}\n MAE = {bootstrap_return['mae']:.2f} """,fontsize=10)

def graph_erreur_vs_aire_plancher(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='sup_planch_tot',y='erreur_bruitee',xlabel='Aire plancher [$m^2$]',ylabel='$y_{{obs}}-y_{{pred}} + \\epsilon$ [pl.]',kind='scatter',ax=ax,ylim=ylim,title="a) Superficie d'étage",label='Données')
    else:    
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='sup_planch_tot',y='error',xlabel='Aire plancher [$m^2$]',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',kind='scatter',ax=ax,ylim=ylim,title="a) Superficie d'étage",label='Données')
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e=+{spot_error}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e=-{spot_error}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)

def graph_erreur_vs_date_constr(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='premiere_constr',y='erreur_bruitee',xlabel='Année de construction [-]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,title='a) Temps',label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='premiere_constr',y='error',xlabel='Année de construction  [-]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,title='a) Temps',label='Données')
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e=+{spot_error}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e=-{spot_error}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)

def graph_erreur_vs_distance_parlement(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='dist_to_parliament',y='erreur_bruitee',xlabel='Distance Parlement [m]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,title='b) Espace',label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='dist_to_parliament',y='error',xlabel='Distance Parlement [m]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,title='b) Espace',label='Données')
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e=+{spot_error}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e=-{spot_error}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)

def graph_erreur_vs_superf_lot(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='superf_lot',y='erreur_bruitee',xlabel='Superficie Lot [$m^2$]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,title="a) Superficie lot vs erreur",label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='superf_lot',y='error',xlabel='Superficie Lot [$m^2$]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,title='a) Superficie lot vs erreur',label='Données')
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e=+{spot_error}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e=-{spot_error}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)


def graph_erreur_vs_n_log(val_data:pd.DataFrame,ax:plt.axes,fig:plt.figure, jitter_bool:bool=False,spot_error:int=None):
    n_tot = len(val_data)
    n_non_null = int(val_data.loc[~val_data['n_logements_tot'].isna()].count().values[0])
    if n_non_null==0:
        ax.axis('off')
        fig.text(0.65,0.5,'Aucune propriété avec logements')
    else:
        if jitter_bool:
            ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
            val_data.plot(x='n_logements_tot',y='erreur_bruitee',xlabel='Nombre de logements [-]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,label='Données')
        else:
            ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
            val_data.plot(x='n_logements_tot',y='error',xlabel='Nombre de logements [-]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,label='Données')
        
        ax.set_title(f'b) Propriétés avec logements $n_{{avec-log}}$ = {n_non_null}')
        ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
        
        if spot_error is not None:
            ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e=+{spot_error}$ pl.')
            ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e=-{spot_error}$ pl.')
            ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)

def graph_erreur_vs_val_role(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='valeur_totale',y='erreur_bruitee',xlabel='Valeur au rôle [\$]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,title='b) Valeur au rôle vs erreur',label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='valeur_totale',y='error',xlabel='Valeur au rôle [\$]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,title='b) Valeur au rôle vs erreur',label='Données')
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--',label=f'$e=+{spot_error}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.',label=f'$e=-{spot_error}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)

def graph_erreur_vs_y_pred(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None,perc_error=None):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data.plot(x='y_pred',y='erreur_bruitee',xlabel='Prédiction [pl.]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,title='b) Prédit vs erreur',label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data.plot(x='y_pred',y='error',xlabel='Prédiction [pl.]',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,title='b) Prédit vs erreur',label='Données')
    ax.axhline(0,color='red',linestyle="--", label=f'$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--', label=f'+{spot_error:.0f} pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.', label=f'-{spot_error:.0f} pl.')
    if perc_error is not None:
        x_max = val_data['y_pred'].max()
        ax.axline((0,0),(x_max,x_max*perc_error),color='#ff7f0e',linestyle='--', label=f'+{(perc_error*100):.0f} \%.')
        ax.axline((0,0),(x_max,-x_max*perc_error),color='#ff7f0e',linestyle='-.', label=f'-{(perc_error*100):.0f} \%.')
    if perc_error is not None or spot_error is not None:
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=8)

def graph_erreur_vs_val_m2(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    val_data_add = val_data.copy()
    val_data_add['val_m2'] = val_data_add['valeur_totale']/val_data_add['superf_lot']
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data_add.plot(x='val_m2',y='erreur_bruitee',xlabel = r"Intensité de développement $\left[\frac{\$}{m_{\mathrm{lot}}^{2}}\right]$",kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,title='a) Intensité développement vs erreur',label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data_add.plot(x='val_m2',y='error',xlabel = r"Intensité de développement $\left[\frac{\$}{m_{\mathrm{lot}}^{2}}\right]$",kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,title='a) Intensité développement vs erreur',label='Données')
    ax.axhline(0,color='red',linestyle="--", label=f'$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--', label=f'$e=+{spot_error:.0f}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.', label=f'$e=-{spot_error:.0f}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)

def graph_erreur_vs_y_pred_m2_par_val(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    val_data_add = val_data.copy()
    val_data_add['predict'] = val_data_add['y_pred']/val_data_add['valeur_totale']*val_data_add['superf_lot']
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data_add.plot(x='predict',y='erreur_bruitee',xlabel=r'Intensité de prédiction  $\left[\frac{y_{pred}\times m^{2}_{\mathrm{lot}}}{\mathrm{\$}}\right]$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$ [pl.]',ax=ax,ylim=ylim,title='b) Intensité prédiction vs erreur',label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data_add.plot(x='predict',y='error',xlabel=r'Intensité de prédiction  $\left[\frac{y_{pred}\times m^{2}_{\mathrm{lot}}}{\mathrm{\$}}\right]$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$ [pl.]',ax=ax,ylim=ylim,title='b) Intensité prédiction vs erreur',label='Données')
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--', label=f'$e=+{spot_error:.0f}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.', label=f'$e=-{spot_error:.0f}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)
        
def graph_erreur_vs_unite_atypiques(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee'])),np.max(np.abs(val_data['erreur_bruitee']))]
        # Make the boxplot
        val_data.boxplot(
            column='erreur_bruitee',
            by='atypical_units',
            ax=ax,
            ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$  [pl.]',
            xlabel='Unités converties?',
            grid=False
        )
    else:
        ylim = [-np.max(np.abs(val_data['error'])),np.max(np.abs(val_data['error']))]
        # Make the boxplot
        val_data.boxplot(
            column='error',
            by='atypical_units',
            ax=ax,
            ylabel='$y_{{obs}}-y_{{pred}}$  [pl.]',
            xlabel='Unités converties?',
            grid=False
        )
    
    ax.set_ylim(ylim[0],ylim[1])
    
    # Count points per group
    counts = val_data['atypical_units'].value_counts().sort_index()

    # Replace x-tick labels with counts
    ax.set_xticklabels([f"{cat}\n(n={counts[cat]})" for cat in counts.index])
    ax.set_title("a) Unité utilisée par règlement")
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--', label=f'$e=+{spot_error:.0f}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.', label=f'$e=-{spot_error:.0f}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)

def graph_erreur_vs_y_pred_par_m2(val_data:pd.DataFrame,ax:plt.axes,jitter_bool:bool=False,spot_error:int=None):
    val_data_add = val_data.copy()
    val_data_add['predict'] = val_data_add['y_pred']/val_data_add['superf_lot']
    if jitter_bool:
        ylim = [-np.max(np.abs(val_data['erreur_bruitee']))*1.25,np.max(np.abs(val_data['erreur_bruitee']))*1.25]
        val_data_add.plot(x='predict',y='erreur_bruitee',xlabel=r'$\frac{y_{pred}}{m^{2}_{\mathrm{lot}}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}+ \\epsilon$',ax=ax,ylim=ylim,label='Données')
    else:
        ylim = [-np.max(np.abs(val_data['error']))*1.25,np.max(np.abs(val_data['error']))*1.25]
        val_data_add.plot(x='predict',y='error',xlabel=r'$\frac{y_{pred}}{m^{2}_{\mathrm{lot}}}$',kind='scatter',ylabel='$y_{{obs}}-y_{{pred}}$',ax=ax,ylim=ylim,label='Données')
    ax.axhline(0,color='red',linestyle="--",label='$e=0$ pl.')
    if spot_error is not None:
        ax.axhline(spot_error, color='#2ca02c', linewidth=1, linestyle='--', label=f'$e=+{spot_error:.0f}$ pl.')
        ax.axhline(-spot_error, color='#2ca02c', linewidth=1, linestyle='-.', label=f'$e=-{spot_error:.0f}$ pl.')
        ax.legend(loc='center left', bbox_to_anchor=(0.75,0.9), frameon=True, fontsize=10)