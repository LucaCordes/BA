__all__ = ['pd', 'np', 'plt', 'pprint', 'color', 're', 'tex2uni', 'integrate', 'curve_fit', 'c', 'stats', 'CubicSpline', 'u', 'ufloat', 'split', 'ev', 'std', 'tag', 'weighted_mean', 'plots_path', 'tables_path', 'plot', 'format_number', 'table', 'sqrt', 'pi', 'abs', 'sin', 'cos', 'tan', 'arccos', 'arcsin', 'arctan', 'exp', 'log', 'deg2rad', 'rad2deg', 'vectorize', 'un', 'unp', 'flat', 'data_path', 'fn', 'language', 'Iterable', 'plotter', "pretty_exponent"]

from numpy import sqrt,pi,abs,sin,cos,tan,arccos,arcsin,arctan,exp,log,deg2rad,rad2deg,vectorize
import numpy as np
from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.labelsize": "large",
        "text.usetex": False,
        "font.size": 13,
    }
)

from collections.abc import Iterable

from pprint import pprint

import ansi.color as color

import tabulate as __tabulate__
__tabulate__.PRESERVE_WHITESPACE = True
from tabulate import tabulate

import re

from typing import List, Tuple

from pylatexenc.latex2text import LatexNodes2Text

def tex2uni(x, all=True):
    """
    string: str     latex notation in it will interpreted to unicode
    all   : bool    if True the whole string will be searched for latex notation, which can have unexpexted effects, if False only text in enclosed by $...$ will be converted
    """
    exp_map = str.maketrans("0123456789-","⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    def replace_exponent(x):
        match = re.search(r"\^\{\+?(-?\d+)\}", x)
        if match:
            exp = match.group(1).translate(exp_map)
            x = re.sub(r"\^\{\+?(-?\d+)\}", exp, x)
            x = replace_exponent(x)
        return x 
    
    x = replace_exponent(x)
    l2t = LatexNodes2Text().latex_to_text
    if all:
        return l2t(x)
    return re.sub(r"(\$.+?\$)", lambda x: l2t(x.group(1)), x)

import os

import scipy.integrate as integrate
from scipy.optimize import curve_fit
import scipy.constants as c
import scipy.stats as stats
import scipy.odr as odr
from scipy.interpolate import CubicSpline

import itertools

flat = lambda x: list(itertools.chain.from_iterable(x))

import uncertainties as un
from uncertainties import ufloat, UFloat
import uncertainties.umath as umath
import uncertainties.unumpy as unp

un.UFloat.__float__ = lambda self: self.n
un.Variable.__float__ = lambda self: self.n

def u(mu, sigma, tag=None):
    return np.vectorize(lambda m, s, tag: un.ufloat(m, s, tag=tag))(mu, sigma, tag)

ev = np.vectorize(
    lambda x: x.n if type(x) == un.Variable or type(x) == un.UFloat else x,
    otypes=[float],
)  # ev(1+/-0.1)=1, ev(1)=[1]
std = np.vectorize(
    lambda x: x.s if type(x) == un.Variable or type(x) == un.UFloat else 0,
    otypes=[float],
)  # std(1+/-0.1)=0.1, std(1)=[0]
split = lambda X: (ev(X), std(X))
scale_std = lambda x, a, tag=None: np.vectorize(u(ev(x), a * std(x), tag=tag if tag else x.tag))
tag = lambda x, tag: u(*split(x), tag=tag)
weighted_mean = lambda x: u(
    np.sum(ev(x) / std(x) ** 2) / np.sum(1 / std(x) ** 2),
    np.sqrt(1 / np.sum(1 / std(x) ** 2)),
)  # sum(mu_i / sigma_i**2) / sum(1/sigma_i**2)

plots_path  = r"/home/home3/institut_thp/lcordes/Bachelor_Thesis/code/plots/"
tables_path = r"/home/home3/institut_thp/lcordes/Bachelor_Thesis/latex/tables/"
language = "en" # de/en
data_path = "./data/"
 
def plot(f):
    def wrapper(*args, **kwargs_):
        savefig, show, legend, close, subplot = None, None, None, False, False
        passed_kwargs = {}
        
        if "ax" in kwargs_:
            plt.sca(kwargs_["ax"])
            kwargs_.pop("ax")
            subplot = True
            
        if "figsize" in kwargs_:
            plt.figure(figsize=kwargs_["figsize"])
            kwargs_.pop("figsize")

        for key, value in kwargs_.items():
            if key == "xlim":
                plt.xlim(*value)
            elif key == "ylim":
                plt.ylim(*value)
            elif key == "xscale":
                plt.xscale(value)
            elif key == "yscale":
                plt.yscale(value)
            elif key == "xlabel":
                plt.xlabel(value)
            elif key == "ylabel":
                plt.ylabel(value)
            elif key == "title":
                plt.title(value)
            elif key == "bftitle":
                plt.title(r"$\mathbf{" + value.replace(" ", r"\ ") + r"}$")
            elif key == "suptitle":
                plt.suptitle(value)
            elif key == "bfsuptitle":
                plt.suptitle(r"$\mathbf{" + value.replace(" ", r"\ ") + r"}$")
            elif key == "supxlabel":
                plt.gcf().supxlabel(value)
            elif key == "supylabel":
                plt.gcf().supylabel(value)
            elif key == "grid":
                plt.grid(linestyle="--", alpha=0.5)
            elif key == "legend":
                legend = value
            elif key in ("savefig", "filename"):
                savefig = value
            elif key == "show":
                show = value
            elif key == "close":
                close = value
            else:
                passed_kwargs[key] = value

        res = f(*args, **passed_kwargs)

        if legend is True:
            plt.legend()
        elif legend is not None:
            plt.legend(loc=legend)

        plt.tight_layout()
        if savefig:
            plt.savefig(plots_path + savefig, bbox_inches="tight")
            close=True

        if show or (show is None and subplot is False):
            plt.show()
            close=True
        
        if close:
            plt.close()

        return res
    
    return wrapper

plotter = lambda **kwargs: plot(lambda: ...)(**{**kwargs, "show": False, "close": False})

def pretty_exponent(x):
    SUPERSCRIPT_FROM = r"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQTUVWXYZ+-=()"
    SUPERSCRIPT_TO = r"⁰¹²³⁴⁵⁶⁷⁸⁹ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᵀᵁⱽᵂˣʸᶻ⁺⁻⁼⁽⁾" 
    exp_map = str.maketrans(SUPERSCRIPT_FROM, SUPERSCRIPT_TO)
    
    SUBSCRIPT_FROM = "0123456789aehijklmnoprstuvx+-=()"
    SUBSCRIPT_TO = "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ₊₋₌₍₎"
    subscript_map = str.maketrans(SUBSCRIPT_FROM, SUBSCRIPT_TO) 
    
    # latex-exponents like ^{+012} oder ^{-03}
    x = re.sub(
        r"\^\{\s*\+?(-?)(0*)(\d+)\s*\}",
        lambda m: (m.group(1) + m.group(3)).translate(exp_map),
        x)
    
    # general latex superscripts like ^{(1)} or ^{a + b}
    x = re.sub(
        rf"\^\{{([{re.escape(SUPERSCRIPT_FROM)}]+)\}}",
        lambda m: m.group(1).translate(exp_map),
        x)
    
    # general latex subscripts like _{(1)} or _{a + b}
    x = re.sub(
        rf"\_\{{([{re.escape(SUBSCRIPT_FROM)}]+)\}}",
        lambda m: m.group(1).translate(subscript_map),
        x)
    
    # exponents like e-13 or e+05
    x = re.sub(
        r"e\+?(-?)0*(\d+)", 
        lambda m: "·10" + (m.group(1) + m.group(2)).translate(exp_map), 
        x)
    
    return x 

def format_number(x, fmt=None, udigits=2, fdigits=3, fformat="nice", latex=False):
    if fmt is not None:
        return x.__format__(fmt)
    elif isinstance(x, str):
        return x
    elif isinstance(x, un.UFloat) or isinstance(x, un.Variable):
        fmt_str = ".{0}u{1}S".format(udigits, "L" if latex else "P")
        res = x.format(fmt_str)
        return "${}$".format(res) if latex else res
    elif isinstance(x, (float, np.floating)):
        if abs(x) > 1e-4 and abs(x) < 1e4:
            num_str = "{{0:.{0}f}}".format(fdigits).format(x)
        else:
            num_str = "{{0:.{0}e}}".format(fdigits).format(x)
        return pretty_exponent(num_str)
    else:
        return str(x)

fn = format_number


def table(data, headers=None, index_column=None, fmt=None, filename=None, caption=None, show=True, show_latex=False, table_option="H", compact=True, head=None, sort_by_col=False, reverse_sort=False, filepath=None, udigits=2, fdigits=3):
    """ filename with .tex ending!
        fmt: len(fmt) == len(data), pad with None if no formatting is necessary
    """
    if not isinstance(data[0],Iterable): data = np.array(data)[:,None]
    
    # throw helpful errors 
    if headers is not None: 
        assert len(data) == len(headers), f"Number of headers ({len(headers)}) is not equal to number of columns ({len(data)})!"
    len_col1 = len(data[0])
    for i, column in enumerate(data):
        if head is None: 
            assert len(column) == len_col1, f"Column 1 does not have the same length as column {i+1} ({len_col1} vs {len(column)})!"
        
    if head: 
        min_len = min([len(col) for col in data])
        head = min(min_len, head)
        data = [col[:head] for col in data]
    data = np.asarray(data, dtype=object)
    
    if sort_by_col:
        idx = np.argsort(data[sort_by_col])[::-1 if reverse_sort else 1]
        data = data[:,idx]
        
    label = filename.split(".")[0] if filename else ""
    
    if index_column is not None: 
        extra_header = None 
        if index_column is True:
            data = np.insert(data, 0, np.arange(1, len(data[0])+1), axis=0)
            extra_header = "$i$"
        elif len(index_column)==len(data[0]):
            data = np.insert(data, 0, index_column, axis=0)
            extra_header = ""
        elif len(index_column)==len(data[0])+1:
            data = np.insert(data, 0, index_column[1:], axis=0)
            extra_header = index_column[0]
        if headers is not None: headers = np.insert(headers, 0, extra_header)
        
    if show:
        data_ = np.transpose([[fn(x, fmt[i] if fmt is not None else None, udigits=udigits, fdigits=fdigits) for x in col] for i,col in enumerate(data)])
        data_[:,0] = [tex2uni(pretty_exponent(x),False) for x in data_[:,0]]
        if headers is not None: 
            headers_ = [color.fx.bold(tex2uni(pretty_exponent(header))) for header in headers]
        
        table_str = tabulate(data_, headers_ if headers is not None else [], "rounded_outline")
        table_len = len(table_str.split("\n")[0])
        if caption: print(color.fg.boldred(f"{f'<{tex2uni(caption)}>':^{table_len}}"))
        print(table_str)
    
    if filename or filepath or show_latex:
        data_ = [[fn(x, fmt[i] if fmt is not None else None, latex=True) for x in col] for i,col in enumerate(data)]
        
        widths = [max(map(len,col)) for col in data_]
        padded_data = [[f"{row:<{width}}" for row in col] for col,width in zip(data_, widths)] 
        
        lines = [" & ".join(row) + r" \\" for row in np.transpose(padded_data)]
        table = "\n".join(lines)
        headers_ = " & ".join(headers) if headers is not None else None
        
        table = r"""\begin{{longtable}}[{table_option}]{{@{{}}{columns}@{{}}}}

\caption{{{caption}}}
\label{{tab:{label}}}\\

\toprule
{headers}
\midrule
\endfirsthead

\multicolumn{{{ncol}}}{{c}}{{{cont_label}}}\\
\midrule
{headers}
\midrule
\endhead

\bottomrule
\multicolumn{{{ncol}}}{{r}}{{{cont_footer}}}\\
\endfoot

\bottomrule
\endlastfoot

{table}

\end{{longtable}}""".format(
    table_option=table_option,
    columns='l' * len(data),
    caption=caption,
    label=Path(filename).stem if filename else Path(filepath).stem if filepath else "<filename>",
    headers=(headers_ + r"\\" if headers_ else ""),
    ncol=len(data),
    cont_label=("" if compact else 
                ("Fortsetzung der Tabelle \\ref{{tab:{}}}".format(label) if language=="de" 
                 else "Continuation of table \\ref{{tab:{}}}".format(label)).format(label)),
    cont_footer=("" if compact else 
                 ("Fortsetzung auf der nächsten Seite..." if language=="de" 
                  else "Continuation on the next page...")),
    table=table
)

    if show_latex: print(table)
    if filename or filepath: 
        filepath = tables_path + filename if filename else plfilepath
        with open(filepath, "w") as file:
                file.writelines(table)
        print(f"table saved to '{filepath}'")
