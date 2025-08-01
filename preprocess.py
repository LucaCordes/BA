import pandas as pd 
import numpy as np
import os 
from ansi import color
from pathlib import Path
from helpers import info

def preprocess_data(sources, destination_folder, filename, split=[60,20,20], report=True):
        """ 
            filename: filename without ending (.h5)
        """
        def calculate_features(source, N=None):
            data = np.asarray(pd.read_hdf(source, key="table", stop=N))
            E, p_x, p_y, p_z = data[:,:-1:4], data[:,1::4], data[:,2::4], data[:,3::4]
            
            mask = E==0
            def nanify(x):
                x[mask] = np.nan 
                return x
                
            def f(E, p_x, p_y, p_z):
                p_T = np.sqrt(p_x**2 + p_y**2)
                phi = np.arctan2(p_y, p_x)
                eta = 1/2 * np.log((E + p_z) / (E - p_z))
                return p_T, phi, eta
            
            _, tot_phi, tot_eta = f(E.sum(1), p_x.sum(1), p_y.sum(1), p_z.sum(1))
            p_T, phi, eta = f(E, p_x, p_y, p_z)
            
            Delta_phi = (tot_phi[...,np.newaxis] - phi + np.pi) % (2*np.pi) - np.pi
            Delta_eta = tot_eta[...,np.newaxis] - eta
            
            return nanify(p_T), nanify(Delta_phi), nanify(Delta_eta)
        
        def get_bins(p_T, percent_inside):
            N = 100_000
            phi_eta_bins = np.linspace(-.8, .8, 30)
            p_T_bins = np.geomspace(np.nanpercentile(p_T[:N], 100-percent_inside),
                                    np.nanmax(p_T), 40) 
            # print(np.nanmin(p_T), np.nanmax(p_T), np.nanpercentile(p_T[:N], 100-percent_inside))
            return p_T_bins, phi_eta_bins, phi_eta_bins
        
        def discretize(x, bins):
            """ 
                x     = [pt, phi, eta]
                bins  = [pt_bins, phi_bins, eta_bins]
                returns [disk_pt, disk_phi, disk_eta]
            """
            mask = np.isnan(x[0])
            res = []
            for x,bins in zip(x,bins):
                x = np.digitize(x,bins).astype(np.int16)
                x[mask] = -1
                res.append(x)
            return np.array(res)
        
        def get_df(p_T, phi, eta):
            stacked = np.stack([p_T, eta, phi], -1,)
            stacked = stacked.reshape((-1, 600))
            cols = [
                item
                for sublist in [f"PT_{i},Eta_{i},Phi_{i}".split(",") for i in range(200)]
                for item in sublist
            ]
            df = pd.DataFrame(stacked, columns=cols)
            return df
        
        def save(disc_data, bins, split, destination_folder, filename): # disc_data = [p_T, phi, eta]
            path = Path(destination_folder)
            if not path.exists():
                os.mkdir(path)
                
            split_indices = np.cumsum(len(disc_data[0]) * np.asarray(split) / 100).astype(int)[:-1]
            split_data = np.split(disc_data, split_indices, axis=1)
            
            for data,label in zip(split_data, ["train", "val", "test"]):
                destination = path.joinpath(label + "_" + filename + ".h5")
                get_df(*data).to_hdf(destination, key="discretized", mode="w")
                
                for data,key in zip([*bins, split], ["pt_bins", "phi_bins", "eta_bins", "split"]):
                    pd.DataFrame(data).to_hdf(destination, key=key, mode="a")
                    
                if report: info(destination)
                    
            prep_dir = path.joinpath("preprocessing_bins")
            if not prep_dir.exists():
                os.mkdir(prep_dir)
            
            for bins, label in zip([*bins, split], ["pt_bins", "phi_bins", "eta_bins", "split"]):
                destination = prep_dir.joinpath(label + "_" + filename)
                np.save(destination, bins)
                
        features = np.hstack([calculate_features(source) for source in sources],)
        
        if report: print("\n"+color.fg.boldblue(f"<populating {destination_folder}, with {len(features[0])} jets total>"))
        
        bins = get_bins(features[0], 99.9)
        disc_data = discretize(features, bins)
        save(disc_data, bins, split, destination_folder, filename)