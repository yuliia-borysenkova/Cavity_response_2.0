from pathlib import Path
#import scienceplots
import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).with_name("style_home.mplstyle")
#STYLE_PATH = Path(__file__).with_name("style_aps2.mplstyle")

def apply_style(style=None):
    if style is None:
        plt.style.use(STYLE_PATH)
    # if style == "science":
    #     plt.style.context(['science', 'high-vis']) #plt.style.context(['science', 'ieee'])
    else:
        plt.style.use(style)

def new_figure(nrows=1, ncols=1, style=None, **kwargs):
    apply_style(style)
    return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)

def save_figure(fig, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)