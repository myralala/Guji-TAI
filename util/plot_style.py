from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from matplotlib import font_manager, pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"


def project_cjk_font_path() -> Path:
    return FONT_PATH


@lru_cache(maxsize=1)
def get_cjk_font_name() -> str:
    font_path = project_cjk_font_path()
    if not font_path.exists():
        return "DejaVu Sans"
    font_manager.fontManager.addfont(str(font_path))
    return font_manager.FontProperties(fname=str(font_path)).get_name()


def get_plot_rc():
    return {
        "font.family": [get_cjk_font_name(), "DejaVu Sans"],
        "axes.unicode_minus": False,
    }


@contextmanager
def chinese_plot_context():
    with plt.rc_context(rc=get_plot_rc()):
        yield
