from typing import Union
import obtain_data as od
import pandas as pd
import matplotlib.pyplot as plt
def linear_plot_prediction(id_categ:Union[int,list[int]]):
    val_data = od.obtain_data()
    val_data_check = val_data.loc[val_data['id_strate']==id_categ].copy()
    input_data = od.obtain_sample_input_data(id_categ)
    strata_desc = od.obtain_strata()
    strate_desc = str(strata_desc.loc[strata_desc['id_strate']== id_categ,'desc_concat'].values[0])
    joined_val_data = val_data_check.merge(input_data,how='left',on='g_no_lot')

    fig,ax =plt.subplots(nrows=1,ncols=2)
    joined_val_data.plot(y='y_obs',x='sup_planch_tot',kind='scatter',title='Stat vs sup_planch',ax=ax[0])
    joined_val_data.plot(y='y_obs',x='valeur_totale',kind='scatter',title='Stat vs val_tot',ax=ax[1])
    fig.suptitle(strate_desc)
    