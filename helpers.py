from tables_plots_helpers import *
import numpy as np
import pandas as pd
import torch 
import re
from pathlib import Path 
from ansi import color
import os 
import sklearn
import functools 
import datetime 
import inspect
import json
from collections.abc import Iterable

torch.multiprocessing.set_sharing_strategy("file_system")

bf = lambda x: "$\\mathbf{" + x.replace(" ", "\\ ") + "}$"

class vdict(dict):
    def __getitem__(self, key):
        if isinstance(key, Iterable) and not isinstance(key, str):
            return np.array([dict.__getitem__(self, k) for k in key])
        return dict.__getitem__(self, key)
    
paths = vdict({
    "ttbar/train": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_train___1Mfromeach_403030.h5"),
    "ttbar/test": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_test___1Mfromeach_403030.h5"),
    "ttbar/val": Path(r"/net/data_ttk/hreyes/OneBin/TTBar_val___1Mfromeach_403030.h5"),
    "ttbar/samples": Path(r"/net/data_ttk/lcordes/TTBar_500k/samples_100k.h5"),
    
    "z/train": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_train___1Mfromeach_403030.h5"),
    "z/test": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_test___1Mfromeach_403030.h5"),
    "z/val": Path(r"/net/data_ttk/hreyes/OneBin/ZJetsToNuNu_val___1Mfromeach_403030.h5"),
    
    "qcd": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd"),
    "qcd/train": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd/train_qcd.h5"),
    "qcd/test": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd/test_qcd.h5"),
    "qcd/val": Path(r"/net/data_ttk/lcordes/redo/semi-visible/qcd/val_qcd.h5"),
    
    "aachen": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen"),
    "aachen/train": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen/train_aachen.h5"),
    "aachen/test": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen/test_aachen.h5"),
    "aachen/val": Path(r"/net/data_ttk/lcordes/redo/semi-visible/aachen/val_aachen.h5")
    })


def reduce(x, length):
    assert len(x)>=length
    idx = np.linspace(0, len(x), length+1, dtype=int)
    out = [np.mean(x[idx[i]: idx[i+1]]) for i in range(len(idx)-1)]
    return np.array(out)


def walk_dir(pattern=r".*model_best\.pt$", dir=r"/net/data_ttk/lcordes/classifier_var_heads"):
    for dirpath, dirnames, filenames in os.walk(dir):
        for dirname in dirnames:
            if re.search(pattern, dirpath + "/" + dirname):
                yield Path(dirpath) / dirname
        for filename in filenames:
            if re.search(pattern, dirpath + "/" + filename):
                yield Path(dirpath) / filename
                
def select_max(dir, prefix=None):
    """selects the file or folder with the maximum global_step (non-recursive!)
    dir: looks here for files/dirs
    prefix: filters for paths with correct prefix. expected paths are: ".../<prefix>_<global_step>_..."
    """
    paths = list(Path(dir).iterdir())
    if prefix: 
        paths = [x for x in paths
                 if prefix==x.name.split("_")[0]]
    assert np.any(paths), f"No files/folders in dir matching prefix '{prefix}'\npaths are: {paths}"
    matches = [re.search(r"\d+", x.name) for x in paths]
    matches = [int(x.group()) if x else np.nan for x in matches]
    return paths[np.nanargmax(matches)]

def get_metrics(filenames, switch=False, signal_eff=0.3):
    """ returns aucs, bg_rejection, accuracy, epochs of best model
    """
    shape = np.shape(filenames)
    filenames = np.reshape(filenames, -1)
    
    epochs = []
    pred_files = []    
    for filename in filenames:
        model = torch.load(filename, "cpu")
        epochs.append(model.global_epoch)
        tests_folder = Path(filename).parent / f"tests/{model.global_step}_{model.global_epoch}"
        pred_files.append(select_max(tests_folder, "predictions"))
        
    aucs, bg_rejection, accuracy = [], [], []
    for pred_file in pred_files:
        data = np.load(pred_file)
        preds = (data["predictions"]).flatten()
        labels = data["labels"].flatten().astype(bool) ^ switch 
        accuracy.append(1 - np.abs(np.round(preds) - labels).sum() / len(labels))

        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, preds)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        
        aucs.append(roc_auc)
        bg_rejection.append(1/fpr[np.argmin(np.abs(tpr - signal_eff))])
    
    return [np.reshape(aucs, shape), 
            np.reshape(bg_rejection, shape), 
            np.reshape(accuracy, shape),
            np.reshape(epochs, shape),
            ]
    
def select_max_auc(pattern="", dir="/net/data_ttk/lcordes/test_heads_and_protocols/Aachen/TTBar_Backbone"):
    pattern = pattern + ".*/model_best\.pt$"
    models = np.array(list(walk_dir(pattern=pattern, dir=dir)))
    aucs = get_metrics(models)[0]
    return models[np.argmax(aucs)]

def select_pred(model):
    model = torch.load(str(model), "cpu")
    return select_max(model.dir / "tests" / f"{model.global_step}_{model.global_epoch}")

def logger(f):
    """
    Decorator that causes all arguments (explicit or implicit) of the function to be logged to args.jsonl, as a list of json objects with a timestamp. The function to be decorated must take an argument dir. 
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        
        log_data = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": {k: str(v) for k, v in bound.arguments.items()}
        }
        
        dir_path = Path(bound.arguments["dir"])
        dir_path.mkdir(exist_ok=True, parents=True)
        
        with open(dir_path / "args.jsonl", "a") as file:
            file.write(json.dumps(log_data) + "\n")
        
        return f(*args, **kwargs)
    return wrapper

def bins2values(binned_data, bins):
    binned_data, bins = np.asarray(binned_data), np.asarray(bins).reshape(-1)
    vals = 1/2 * np.array([np.nan, *(bins[:-1] + bins[1:]), np.nan])
    return vals[binned_data.reshape(-1)].reshape(binned_data.shape)

def denan(x):
    x = np.asarray(x)
    return x[x!=np.nan]

def load_data(path, N):
    data = []
    for i,key in enumerate(["discretized", "pt_bins", "phi_bins", "eta_bins"]):
        x = pd.read_hdf(path, key=key, stop=N if i==0 else None).to_numpy()
        if i>0: x = x.flatten()
        data.append(x)
    return data[0][:,::3], data[0][:,1::3], data[0][:,2::3], *data[1:]

def load_data2(path, N):
    data = np.asarray(pd.read_hdf(path, "discretized", stop=N))
    return [data[:,::3], data[:,1::3], data[:,2::3], 
            np.exp(np.load("/net/data_ttk/hreyes/OneBin/preprocessing_bins/pt_bins_1Mfromeach_403030.npy")),
            np.linspace(-.8,.8,30), np.linspace(-.8,.8,30)]
    
def lim(x, offset=.05):
    l = np.array((np.nanmin(x), np.nanmax(x)))
    l += offset * np.diff(l) * [-1,1]
    return l 

def info(filepaths, N=6, preview=True):
    """ 
    Prints a table of relevant information of a .h5 file. 
    Information includes: all keys, shape, column, values (preview)
    """
    for filepath in np.atleast_1d(filepaths):
        size_MB = os.path.getsize(filepath) / 1024**2
        if preview:
            try:
                keys, shapes, columns, values = [], [], [], []
                store = pd.HDFStore(filepath, mode='r',)
                for key in store.keys():
                    df = store.get(key)
                    keys.append(key)
                    shapes.append(df.shape)
                    columns.append(list(df.columns))
                    if df.empty:
                        values.append(["<empty>"])
                    elif df.shape[1] == 1:
                        values.append(list(df.iloc[:min(N+1, df.shape[0]), 0]))
                    else:
                        values.append(list(df.iloc[0, :min(N+1, df.shape[1])]))
                store.close()
                
                columns = [" ".join(np.vectorize(str)([*(x[:N]), "..."] if len(x)>N else x)) for x in columns]
                values = [" ".join([*x[:N], "..."] if len(x)>N else x) for x in [np.vectorize(fn)(v) for v in values]]
                table([keys, shapes, columns, values],
                    ["keys", "shape", "columns", "values"],
                    True, caption=f"info for {filepath} ({fn(size_MB)} MB)")
                return
            except:
                ...
        data = np.asarray(pd.read_hdf(filepath, key="table"))
        print(f"{color.fg.boldred(f'<info for {filepath} ({fn(size_MB)} MB)>')}\nshape of \\table: {data.shape}\n")
    

def join_datasets(sources, destination_folder, split=(60,20,20)):
    df = pd.DataFrame()
    for path in sources:
        df = pd.concat([df, pd.read_hdf(path, key="discretized")])
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    split_indices = np.cumsum(len(df) * np.asarray(split) / 100).astype(int)[:-1]
    dfs_split = np.split(df, split_indices)
    
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(exist_ok=True)
    
    for df,label in zip(dfs_split, ["train", "val", "test"]):
        destination = destination_folder.joinpath(label + ".h5")
        df.to_hdf(destination, key="discretized", mode="w")

import scipy
def odr(x, y, p0=[1,0]):
    def lin_func(B, x):
        return B[0] * x + B[1]

    model = scipy.odr.Model(lin_func)
    data = scipy.odr.RealData(x, y)
    odr = scipy.odr.ODR(data, model, beta0=p0)
    
    out = odr.run()
    return out.beta

def get_metrics_table(model_dirs, heads, training_procedures, app_to_caption="", filestem=None, signal_eff=0.3):
    """ filenames_ij = model with i-th head and j-th training protocol
        which = 0 | 1 | 2 corresponding to auc | bg_rejection | accuracy
    """
    scale_sigma = lambda x, lamb: u(ev(x), lamb * std(x))
    model_dirs = np.asarray(model_dirs, dtype=object)
    metric_names = {0: "AUCs",
                    1: "Background rejections",
                    2: "Accuracys",
                    3: "Epoch of best models"}
    
    metrics = [get_metrics(model_dirs + s + "/model_best.pt", signal_eff=signal_eff) for s in ["_1", "_2", "_3"]]
    metrics = u(np.average(metrics, axis=0), 
                np.std(metrics, axis=0, ddof=1))
    
    def augment_avg(data):
        data = np.asarray(data)
        res = np.zeros(np.add(data.shape, 1), dtype=object)
        res[:-1, :-1] = data
        res[:-1,  -1] = data.mean(1)
        res[ -1, :-1] = data.mean(0)
        res[ -1,  -1] = 0
        return res
    
    for which in range(4):
        data = augment_avg(metrics[which])
        header = [*heads, "Average"]
        index_column = [*[x.replace("_","") for x in training_procedures], "Average"]
        fmt = None #[None, *([metric_fmts[which]] * (len(heads) + 1))]
        
        table(data=data, 
            headers=header,
            index_column=index_column,
            caption=metric_names[which] + app_to_caption,
            fmt=fmt,
            filename=filestem + f"_{which}.tex" if filestem else None,
            udigits=2)