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


def no_axis(x=False, y=False):
    ax = plt.gca()# if fig is None else fig.gca()
    if not x: ax.axes.xaxis.set_ticks([])
    if not y: ax.axes.yaxis.set_ticks([])
    return ax
