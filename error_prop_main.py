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
    srh.single_strata(36,[0,10],5)
    #srh.single_strata(43,[0,20],5)
    #srh.single_strata(44,[0,100],5)
    #srh.single_strata(40,[0,400],5)
    #srh.single_strata(41,[0,400],5)
    #srh.single_strata(39,[0,200],5)
    #srh.single_strata(42,[0,200],5)
    #srh.single_strata(34,[0,400],5)
    plt.show()