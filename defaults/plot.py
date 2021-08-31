import matplotlib as mpl
import matplotlib.pyplot as plt

dpi = 300

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

TUMBlue = "#0065BD"
TUMAccentOrange = "#E37222"
TUMAccentGreen = "#A2AD00"
TUMSecondaryBlue2 = "#003359"
TUMDarkGray = "#333333"

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[TUMBlue, TUMAccentOrange, TUMAccentGreen, TUMSecondaryBlue2, TUMDarkGray])

def enable_latex_export():
    width = 1.8
    plt.rcParams.update({'text.usetex': True,
                         'pgf.texsystem': 'pdflatex',
                         'text.latex.preamble': r'\usepackage[T1]{fontenc} \usepackage[sc]{mathpazo}',
                         'font.family': 'serif',
                         'figure.figsize': (width, width * 0.7),
                         'font.size': 8.35,
                         'figure.titlesize': 10.95,
                         'axes.titlesize': 10.95,
                         'axes.labelsize': 10.95,
                         'legend.fontsize': 10.95,
                         'axes.linewidth': 0.5,
                         'xtick.major.width': 0.5,
                         'ytick.major.width': 0.5,
                         'lines.linewidth': 1,
                         })

def no_axis(x=False, y=False):
    ax = plt.gca()# if fig is None else fig.gca()
    if not x: ax.axes.xaxis.set_ticks([])
    if not y: ax.axes.yaxis.set_ticks([])
    return ax
