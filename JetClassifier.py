from tables_plots_helpers import *
from helpers import *
from JetTransformer import JetTransformer
from Heads import *

import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import torch 
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os 
from pathlib import Path
import datetime 
from tabulate import tabulate
import time
from tqdm.auto import tqdm 
import contextlib


@plot
def __plot_preds__(paths, labels):
    for i, [path, label] in enumerate(zip(np.atleast_1d(paths), np.atleast_1d(labels))):
        data = np.load(path)
        logits = (data["logits"]).flatten()
        labels = data["labels"].flatten().astype(bool)
        
        bins = np.linspace(*lim(logits), 100)
        for logits in [logits[labels], logits[~labels]]:
            values, bins = np.histogram(logits, bins, density=True)
            plt.stairs(values, bins, color=f"C{i}")
        
    plotter(grid=1, xlabel="logits", ylabel="density")

@plot 
def __plot_roc__(filenames, labels, linear=False, switch=None, signal_eff=0.3):
    aucs, bg_rejection = [], []
    for filename,label,switch in zip(filenames, labels, switch):
        data = np.load(filename)
        preds = (data["predictions"]).flatten()
        labels = data["labels"].flatten().astype(bool)

        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, preds)
        if switch: fpr, tpr = tpr, fpr
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        
        aucs.append(roc_auc)
        bg_rejection.append(1/fpr[np.argmin(np.abs(tpr - signal_eff))])

        plt.plot(tpr, fpr if linear else 1/fpr, label=f"{label} (AUC = {roc_auc:.5f})")
        
    if linear:
        plt.plot(x:=[0,1], x, 'k--', label="random (AUC=0.5)")
        plotter(ylabel=r"$\epsilon_B$", grid=1, ylim=(0,1))
    else:
        plt.plot(x:=np.linspace(1e-7,1,1000), 1/x, 'k--', label="random (AUC=0.5)")
        plt.grid(axis="y",which="both", ls="-", alpha=0.5)
        plt.grid(axis="x", ls="-", alpha=0.5)
        plotter(ylabel=r"$1/\epsilon_B$", ylim=(1,1e4), yscale="log")
        
    plt.xticks(np.linspace(0,1,11))
    plotter(xlabel=r"$\epsilon_S$", xlim=[0,1])
    
    return aucs, bg_rejection

def __plot_classifiers__(filenames, labels, switch=None):
    filenames, labels = np.atleast_1d(filenames), np.atleast_1d(labels)
    if switch is None: switch = np.zeros_like(filenames, dtype=bool)
    
    fig, ax = plt.subplots(2,1, figsize=(9,7))
    
    __plot_preds__(filenames, labels, title=r"Activations for Signal and Background", ax=ax[0])
    aucs, bg_rejection = __plot_roc__(filenames, labels, title=r"ROC", switch=switch, ax=ax[1])
    
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    fig.text(.5, 1.1, "Classifier Performances", ha="center", fontsize=12, fontweight="bold")
    
    table([labels, aucs, bg_rejection],
          ["Classifier", "AUC", r"$1/\epsilon_B (\epsilon_S=0.3)$"], 
          fmt=[None, ".5f", ".1f"])
    
    return aucs

            
class JetClassifier(nn.Module):
    """ 
        dir structure: 
            - <dir>
                - model_best.pt
                - model_last.pt
                - checkpoints
                    - <global_step>_<global_epoch>[_<postfix>].pt
                - tests
                    - <global_step>_<global_epoch>
                        - predictions_<num_events>.npz
                        - classifier_performance_<num_events>.pdf
                    - ...
                - logs
                    - log1
                    - ...
    """
    def __init__(self, dir, backbone, head, bg_file, sig_file, cutoff=None, exist_ok=False):
        super().__init__()
        self.init_backbone = str(backbone)
        self.init_head = str(head)
        
        if isinstance(backbone, str): 
            backbone = torch.load(backbone, map_location=torch.device("cpu"))
        elif backbone is None:
            backbone = JetTransformer()
        else: assert isinstance(backbone, nn.Module), f"invalid backbone: {backbone}"
        self.feature_embeddings = backbone.feature_embeddings
        self.layers = backbone.layers[:(-cutoff if cutoff else None)]
        self.out_norm = backbone.out_norm
        self.dropout = backbone.dropout
        
        if isinstance(head, nn.Module):
            self.head = head
        elif isinstance(head, str): 
            self.head = torch.load(head, map_location=torch.device("cpu")).head
        elif head==None:
            self.head = NormalHead()
        else: assert False, f"invalid head: {head}"
            
        self.global_step = 0
        self.global_epoch = 0
        self.total_time_trained = 0 # seconds
        self.total_time_tested = 0 # seconds
        
        self.train_loss = [np.inf]
        self.val_loss = [np.inf]
        self.auc = [0]
        
        self.bg_file = bg_file
        self.sig_file = sig_file
        self.dir = Path(dir)
        
        self.num_const = 128
        self.num_features = 3
        
        self.dir.mkdir(exist_ok=exist_ok)
        (self.dir / "logs").mkdir(exist_ok=exist_ok)
        (self.dir / "tests").mkdir(exist_ok=exist_ok)
        (self.dir / "checkpoints").mkdir(exist_ok=exist_ok)
        
        self.info("status", log=True, print=True)

    def set_dropout_p(self, p):
        p = float(p)
        self.dropout_p = p
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = p

    def migrate(self, new_dir):
        old_dir = self.dird
        self.dir = Path(new_dir)
        
        old_dir.rename(new_dir)

    def save(self, name="model_last"):
        torch.save(self, self.dir / f"{name}.pt")
        
    def checkpoint(self, postfix=None):
        path = self.dir / "checkpoints" / f"{self.global_step}_{self.global_epoch}{'_' + postfix if postfix else ''}.pt"
        torch.save(self, path)
    
    def logger(self):
        return SummaryWriter(self.dir / "logs", purge_step=self.global_step)
        
    def info(self, which="all", log=False, print=True): 
        """ 
            which = "all" / "progress" / "status" 
        """
        which = np.atleast_1d(which)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        progress = [["min_train_loss:", min(self.train_loss)],
                    ["min_val_loss:", min(self.val_loss)],
                    ["best_auc:", min(self.auc)],
                    ["total_time_trained:", datetime.timedelta(seconds=int(self.total_time_trained))],
                    ["total_time_tested:", datetime.timedelta(seconds=int(self.total_time_tested))],
                    ["global_epoch:", self.global_epoch],
                    ["global_step:", self.global_step]]
        
        status = [["device:", device], 
                  ["dir:", self.dir],
                  ["bg_file:", self.bg_file],
                  ["sig_file:", self.sig_file],
                  ["num params:", sum(p.numel() for p in self.parameters())],
                  ["num trainable params:", sum(p.numel() for p in self.parameters() if p.requires_grad)]]
        
        if which=="progress":
            data = progress
        elif which=="status":
            data = status
        elif which=="all":
            data = [*progress, *status]
                    
        if print: 
            table(np.transpose(data), caption=f"info for {self.dir.name}")
        
        if log:
            self.logger().add_text("Info", tabulate(data, tablefmt="html"), global_step=max(1, self.global_step))

    def set_freeze(self, freeze):
        self.freeze = freeze
        for layer in [self.feature_embeddings, self.layers, self.out_norm]:
            for p in layer.parameters():
                p.requires_grad = not freeze 
    
    def get_dataloader(
        self,
        num_events=1000_000,
        batch_size=100,
        num_workers=4,
        train=True,
        split=(60, 20) # train / val
    ):
        """ 
        num_const: jets from bg and sig EACH! Total amount of jets = 2 * num_cons
        bg_file and sig_file: refer to the data used for training. validation and testing data paths are infered by replacing "train" to "val" or "test" in the filename
        val_split: Training data has size 2*num_events, val data has size split[1] / split[0] * 2 * num_events, and the test data has size 2 * num_events if selected using train=False
        """
        def __load_data__(file, stop, num_const):
            """ 
                returns num_events × num_const × 3 (pt/eta/phi)
            """
            data = pd.read_hdf(file, key="discretized", stop=stop)
            data = data.to_numpy(dtype=np.int64)[:, : num_const * 3]
            data = data.reshape(data.shape[0], -1, 3)
            if stop is not None:
                assert len(data)==stop, f"data source '{file}' contains only {len(data)} events, but {stop} where requested!"
            return data

        def __get_dataloader__(bg_file, sig_file, num_events, num_const, batch_size, num_workers, shuffle=True):
            bg = __load_data__(bg_file, num_events, num_const)
            sig = __load_data__(sig_file, num_events, num_const)
            if num_events is None: 
                num_events = min(len(bg), len(sig))
                print(f"loaded {num_events} events from bg and sig each. bg has {len(bg)} total, sig has {len(sig)} total.")
                bg = bg[:num_events]
                sig = sig[:num_events]

            data = torch.from_numpy(np.concatenate((bg, sig), 0))
            label = torch.from_numpy(np.append(np.zeros(len(bg)), np.ones(len(sig))))
            padding = data[:, :, 0] < 0

            dataset = TensorDataset(
                data,
                padding.bool(),
                label,
            )
            loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
            return loader
        
        bg_file, sig_file = Path(self.bg_file), Path(self.sig_file)
        val_bg_file       = bg_file.with_name( bg_file.name.replace("train", "val") )
        val_sig_file      = sig_file.with_name( sig_file.name.replace("train", "val") )
        test_bg_file      = bg_file.with_name( bg_file.name.replace("train", "test") ) 
        test_sig_file     = sig_file.with_name( sig_file.name.replace("train", "test") )
        
        
        num_events_val = int(num_events * (split[1] / split[0])) if num_events else None        
        # num_events_test = int(num_events * (split[2] / split[0])) if num_events else None        
        
        if train: 
            train_dataloader =  __get_dataloader__(bg_file, sig_file, num_events, self.num_const, batch_size, num_workers)
            val_dataloader =  __get_dataloader__(val_bg_file, val_sig_file, num_events_val, self.num_const, batch_size, num_workers)
            return  train_dataloader, val_dataloader
        else: 
            test_dataloader = __get_dataloader__(test_bg_file, test_sig_file, num_events, self.num_const, batch_size, num_workers)
            return test_dataloader
    
    def forward(self, jet, padding):
        padding = padding.to(dtype=torch.bool)
        
        seq_len = jet.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=jet.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        
        jet[padding] = 0 

        emb = self.feature_embeddings[0](jet[..., 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](jet[..., i])

        for layer in self.layers:
            emb = layer(emb, src_key_padding_mask=padding, src_mask=causal_mask)

        emb = self.out_norm(emb)
        emb = self.dropout(emb)

        return self.head(emb, padding)

    def train_model(self, epochs=10, num_events=1_000_000, lr=1e-3, min_lr=1e-6, weight_decay=1e-5, batch_size=100, num_workers=1, use_profiler=False, num_events_test=100_000, testing_steps=1, logging_steps=100, dropout_p=0.1, checkpoint=True, freeze=False, scheduler="CosineAnnealingLR"):
        """
            scheduler: CosineAnnealingLR / Constant
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        log_data = [["device:", device],
                    ["epochs:", epochs],
                    ["num_events:", num_events],
                    ["freeze:", freeze],
                    ["scheduler:", scheduler],
                    ["lr:", lr],
                    ["min_lr:", min_lr],
                    ["dropout_p:", dropout_p],
                    ["weight_decay:", weight_decay],
                    ["batch_size:", batch_size],
                    ["num_workers:", num_workers],
                    ["use_profiler:", use_profiler],
                    ["num_events_test:", num_events_test],
                    ["testing_steps:", testing_steps],
                    ["logging_steps:", logging_steps]]
        
        logger = self.logger()
        logger.add_text("Train/Arguments", tabulate(log_data, tablefmt="html"), global_step=self.global_step)
    
        self.set_dropout_p(dropout_p)
        self.set_freeze(freeze)
        if self.global_step==0 and num_events_test: self.test_model(num_events_test, batch_size, num_workers, log=True)
        
        train_loader, val_loader = self.get_dataloader(num_events, batch_size, num_workers, train=True,)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)
        
        if scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * epochs, eta_min=min_lr)
        elif scheduler == "Constant":
            class ConstantLRScheduler:
                def __init__(self, optimizer, lr):
                    self.optimizer = optimizer
                    self.lr = lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr

                def step(self):
                    pass

                def get_last_lr(self):
                    return [self.lr]
            
            scheduler = ConstantLRScheduler(optimizer, lr)
            
        else: assert False, f"invalid scheduler: '{scheduler}'"
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 
        #     T_0=int(2*num_events/batch_size * epochs + 1),
        #     eta_min=1e-6,
        #     )
        scaler = torch.cuda.amp.GradScaler() if device=="cuda" else torch.cpu.amp.GradScaler()
        criterion = nn.BCEWithLogitsLoss()

        profiler_context = (
            torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.dir / f"logs/profiler_{self.global_step}")),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) if use_profiler else contextlib.nullcontext()
        )

        with profiler_context as profiler:
            for epoch in tqdm(range(epochs), desc="epochs           ", position=0, colour="blue"):
                t0 = time.time()
                
                self.train()
                loss_list = []
                for jet, padding, label in tqdm(train_loader, desc="training batches  ", position=1, leave=False, colour="green"):
                    self.global_step += 1
                    jet, padding, label = jet.to(device), padding.to(device), label.to(device)

                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast() if device=="cuda" else torch.cpu.amp.autocast():
                        output = self(jet, padding).squeeze(1)
                        loss = criterion(output, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    loss_list.append(loss.item())
                    if (self.global_step+1) % logging_steps == 0:
                        train_loss = np.mean(loss_list)
                        self.train_loss.append(train_loss)
                        logger.add_scalar("Train/Loss", train_loss, self.global_step)
                        loss_list = []
                        
                    scheduler.step()
                    logger.add_scalar("Train/LR", scheduler.get_last_lr()[0], self.global_step)
                    
                    if use_profiler:
                        profiler.step()
                        
                self.eval()
                val_loss_list = []
                # acc_list = []
                with torch.no_grad():
                    for jet, padding, label in tqdm(val_loader, desc="validation batches", position=1, leave=False, colour="red"):
                        jet, padding, label = jet.to(device), padding.to(device), label.to(device)
                        output = self(jet, padding).squeeze(1)
                        loss = criterion(output, label).item()
                        # acc = torch.sum((output>0) == label)
                        val_loss_list.append(loss)

                self.global_epoch += 1
                self.total_time_trained += time.time() - t0
                
                val_loss = np.mean(val_loss_list)
                self.val_loss.append(val_loss)
                if val_loss < min(self.val_loss):
                    torch.save(self, self.dir / "model_best.pt")
                logger.add_scalar("Val/Loss", val_loss, self.global_step)
                
                torch.save(self, self.dir / "model_last.pt")
                
                if num_events_test and epoch%testing_steps == 0: 
                    self.test_model(num_events=num_events_test, batch_size=batch_size, num_workers=num_workers, log=True)
                
        if checkpoint: self.checkpoint()
        self.info(which="all", log=True, print=True)
        del train_loader, val_loader
      
    def test_model(self, num_events=100_000, batch_size=100, num_workers=1 , save_pdf=True, log=False):
        t0 = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device {device}")
        self.to(device)
        
        self.eval()
        test_loader = self.get_dataloader(num_events, batch_size, num_workers, train=False)
        
        labels, logits, predictions = [], [], []
        with torch.no_grad():
            for jet, padding, label in tqdm(test_loader, desc="testing"):
                jet, padding, label = jet.to(device), padding.to(device), label.to(device)
                
                logit = self(jet, padding).squeeze(1)
                
                labels.append(label.cpu().numpy())
                logits.append(logit.cpu().numpy())
                predictions.append(torch.sigmoid(logit).cpu().numpy())
        
        del test_loader
            
        labels = np.concatenate(labels)
        logits = np.concatenate(logits)
        predictions = np.concatenate(predictions)
        
        path = self.dir / "tests" / f"{self.global_step}_{self.global_epoch}"
        path.mkdir(exist_ok=True)
        predictions_path = path / f"predictions_{num_events}"
        np.savez(predictions_path, labels=labels, logits=logits, predictions=predictions)
        
        # plot 
        paths, labels = [str(predictions_path) + ".npz"], [self.dir.name]
        fig, ax = plt.subplots(2,1, figsize=(9,7))
        
        __plot_preds__(paths, labels, title=r"Activations for Signal and Background", ax=ax[0])
        aucs, bg_rejection = __plot_roc__(paths, labels, title=r"ROC", switch=[0], ax=ax[1])
        fig.suptitle(f"{self.global_epoch}. epoch, AUC = {aucs[0]:.5}, $\\mathbf{{1/\\epsilon_B(\\epsilon_S=0.3)}}$ = {bg_rejection[0]:.1f}", ha="center", fontsize=12, fontweight="bold")
        fig.tight_layout()
        
        if save_pdf: plt.savefig(path / f"classifier_performance_{num_events}.pdf", bbox_inches="tight")

        if log:
            logger = self.logger()
            logger.add_figure("Test/ROC", plt.gcf(), max(1, self.global_step), False)
            logger.add_scalar("Test/AUC", aucs[0], max(1, self.global_step))
        
        table([labels, aucs, bg_rejection],
            ["Classifier", "AUC", r"$1/\epsilon_B (\epsilon_S=0.3)$"], 
            fmt=[None, ".5f", ".1f"])
        
        self.auc.append(aucs[0])
        self.total_time_tested += time.time() - t0

    def test_gradients(self, n=1):
        """
            n: computes the gradients of n signal- and n bg-jets
            returns: gradients filepath = (<model_dir>/tests/<global_step>_<global_epoch>/gradients_<n>.npz) 
            
            gradients file has keys: 
                jets (2*n, num_const, feature), 
                gradients (2*n, num_const, feature),
                paddings (2*n, num_const), 
                labels (2*n)
        """
        def sort_jet(x):
            _, idx = x[:,:,0].sort(dim=1, descending=True)
            return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(-1,-1,x.size(2)))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.to(device)
        self.eval()

        test_loader = list(self.get_dataloader(n, batch_size=1, num_workers=0, train=False))
        np.random.shuffle(test_loader)

        jets, paddings, labels, gradients = [], [], [], []

        for jet, padding, label in tqdm(test_loader):
            jet = jet.to(device)
            padding = padding.to(device)
            label = label.to(device)

            y = self(jet, padding)

            gradient = torch.zeros_like(jet, dtype=torch.float, device=device)

            for const in range(jet.shape[-2]):
                if padding[0, const]: continue  # Skip padded

                for feature in range(jet.shape[-1]):
                    jet_prime = jet.detach().clone()

                    sign = +1
                    value = jet[0, const, feature]
                    if (feature == 0 and value == 40) or (feature != 0 and value == 30):
                        jet_prime[0, const, feature] -= 1
                        sign = -1
                    else:
                        jet_prime[0, const, feature] += 1

                    jet_prime = sort_jet(jet_prime)

                    with torch.no_grad():
                        y_prime = self(jet_prime, padding)

                    gradient[0, const, feature] = sign * (y - y_prime).item()

            jets.append(jet[0].cpu().numpy())
            paddings.append(padding[0].cpu().numpy())
            labels.append(label[0].cpu().numpy())
            gradients.append(gradient[0].cpu().numpy())

        jets = np.asarray(jets)
        paddings = np.asarray(paddings)
        labels = np.asarray(labels)
        gradients = np.asarray(gradients)

        filename = f"{self.dir}/tests/{self.global_step}_{self.global_epoch}/gradients_{n}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, jets=jets, paddings=paddings, labels=labels, gradients=gradients)
        print(f"Gradients saved as: '{filename}.npz'")
        
        return f"{filename}.npz"