from pathlib import Path

import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).with_name("style.mplstyle")

def apply_style(style=None):
    if style is None:
        plt.style.use(STYLE_PATH)
    else:
        plt.style.use(style)

def new_figure(style=None, nrows=1, ncols=1, **kwargs):
    apply_style(style)
    return plt.subplots(nrows=nrows, ncols=ncols, **kwargs)

def save_figure(fig, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)