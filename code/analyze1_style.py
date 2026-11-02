import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.6,
                     'savefig.dpi': 300, 'legend.frameon': False, 'axes.axisbelow': True})
COL = {'GAMA': '#2a78d6', 'LR': '#8a8984', 'LightGBM': '#eb6834', 'EBM': '#1baf7a', 'LightGBM-mono': '#eda100',
       'GAM (unconstrained)': '#6250d6', 'GAMA (quantile knots)': '#e87ba4', 'GAMA (+ interactions)': '#008300'}
