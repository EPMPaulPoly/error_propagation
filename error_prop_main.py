import stats_routine_helper as srh
import matplotlib.pyplot as plt
if __name__ =="__main__":
    #data =srh.stats_routine()
    #print(data)
    # --- Matplotlib style for LaTeX / Latin Modern ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.labelsize": 10,
        "font.size": 12,
        "legend.fontsize": 10,  # Increased legend font size
        "axes.titlesize":10,
        "figure.autolayout": True,
        "xtick.labelsize": 10,  # x-axis tick labels
        "ytick.labelsize": 10,  # y-axis tick labels
        "figure.autolayout": False
    })
    n_it_range= [1000,10000]
    n_it_select = 2000
    # Res
    srh.single_strata(36,10,jitter=0.3,spot_error=2,interval_plots=True,error_plots=True)#
    plt.close()
    srh.single_strata(43,10,jitter=0.3,interval_plots=True,error_plots=True,spot_error=2)#,jitter=0.3
    plt.close()
    srh.single_strata(44,10,interval_plots=True,error_plots=True,spot_error=5,perc_error=0.2)
    plt.close()
    # comm
    srh.single_strata(40,10,error_plots=True,spot_error=10,perc_error=0.2,unit_plots=True)
    plt.close()
    srh.single_strata(40,10,max_error=75,interval_plots=True,spot_error=10,perc_error=0.2)#
    plt.close()
    # serv
    srh.single_strata(41,10,error_plots=True,spot_error=10,perc_error=0.2,unit_plots=True)#,max_error=50
    plt.close()
    srh.single_strata(41,10,max_error=50,interval_plots=True,spot_error=10,perc_error=0.2)#,max_error=50
    plt.close()
    #ind
    srh.single_strata(39,10,spot_error=10,perc_error=0.2,error_plots=True,unit_plots=True)
    plt.close()
    srh.single_strata(39,10,spot_error=10,perc_error=0.2,max_error=75,error_plots=True)
    plt.close()
    srh.single_strata(39,10,spot_error=10,perc_error=0.2,max_error=75,interval_plots=True)#
    plt.close()

    # Assemblée récréation
    srh.single_strata(42,10,spot_error=10,perc_error=0.2,error_plots=True,unit_plots=True)
    plt.close()
    srh.single_strata(42,10,spot_error=10,perc_error=0.2,max_error=100,interval_plots=True)
    plt.close()
    #,max_error=100
    #usage multiple
    srh.single_strata(34,10,max_error=100,interval_plots=True,spot_error=10,perc_error=0.2)#,max_error=100
    plt.close()
    srh.single_strata(34,10,error_plots=True,spot_error=10,perc_error=0.2,unit_plots=True)
    plt.close()
    #plt.show()