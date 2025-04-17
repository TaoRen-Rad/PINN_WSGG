from matplotlib import pyplot as plt

def setup():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 10

    rc = {
        'xtick.major.size': 2,
        'xtick.major.width': 0.5,
        'ytick.major.size': 2,
        'ytick.major.width': 0.5,
        'xtick.bottom': True,
        'ytick.left': True,
        'font.size': MEDIUM_SIZE,
        'axes.titlesize': MEDIUM_SIZE,
        'axes.labelsize': MEDIUM_SIZE,
        'xtick.labelsize': SMALL_SIZE,
        'ytick.labelsize': SMALL_SIZE,
        'legend.fontsize': SMALL_SIZE,
        'figure.titlesize': BIGGER_SIZE,
        'savefig.dpi': 300,
        'figure.dpi': 300,
        "font.family": "serif",
        "text.usetex": True,
        "font.serif": ["Liberation Serif", "DejaVu Serif", 
                        "Nimbus Roman No9 L", "Times"]
    }

    plt.rcParams.update(rc)