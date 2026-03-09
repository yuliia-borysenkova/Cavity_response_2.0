from pathlib import Path

import matplotlib.pyplot as plt

STYLE_PATH = Path(__file__).with_name("style.mplstyle")

def apply_style():
    plt.style.use(STYLE_PATH)

def new_figure(**kwargs):
    apply_style()
    return plt.subplots(**kwargs)

def save_figure(fig, path, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)