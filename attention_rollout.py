from Heads import CLSTokenHead

import numpy as np
import torch 
import os
from tqdm import tqdm

def patch_transformer(transformer):
    def patch_attention(m):
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False

            return forward_orig(*args, **kwargs)

        m.forward = wrap

    class SaveOutput:
        def __init__(self):
            self.outputs = [] # [jet, heads, seq_len, seq_len]

        def __call__(self, module, module_in, module_out):
            self.outputs.append(module_out[1][0].detach())

        def clear(self):
            self.outputs = []
            
    save_outputs = []
    for i in range(len(transformer.layers)):
        save_outputs.append(SaveOutput())
        patch_attention(transformer.layers[i].self_attn)
        transformer.layers[i].self_attn.register_forward_hook(save_outputs[-1])
        
    if isinstance(transformer.head, CLSTokenHead):
        for i in range(len(transformer.head.transformer.layers)):
            save_outputs.append(SaveOutput())
            patch_attention(transformer.head.transformer.layers[i].self_attn)
            transformer.head.transformer.layers[i].self_attn.register_forward_hook(save_outputs[-1])
        
    return save_outputs

def get_attn(model, num_events = 100):
    """ attn.shape = [layer, jet, heads, seq_len, seq_len]
        jets = [jet, const, feature]
        paddings = [jet, const]
        labels = [jet]
    """
    model = torch.load(model, "cpu")

    save_outputs = patch_transformer(model)
    jets, paddings, labels = [], [], []
    for jet, padding, label in tqdm(
        model.get_dataloader(num_events=num_events,batch_size=1,train=False,  num_workers=0)
        ):
        jets.append(jet[0].numpy())
        paddings.append(padding[0].numpy())
        labels.append(label[0].numpy())
        _ = model(jet, padding)
        
    attn = [x.outputs for x in save_outputs]
    return attn, jets, paddings, labels

def augment_attn(attn, bb_layers=8):
    augmented_attn = np.zeros((len(attn), 
                               len(attn[0]), 
                               len(attn[0][0]),
                               len(attn[0][0][0])+1, 
                               len(attn[0][0][0])+1))
    augmented_attn[..., 0, 0] = 1
    augmented_attn[:bb_layers, ..., 1:, 1:] = attn[:bb_layers]
    augmented_attn[bb_layers:, ...] = attn[bb_layers:]
    
    return augmented_attn

def get_attn_rollouts(attn, paddings, alpha=0.2):
    """ rollouts: [jet, seq_len, seq_len]
    """
    head_avg = attn.mean(2)   # [layers, jet, seq_len, seq_len]
    n_const = (~np.array(paddings)).sum(-1).sum(-1)

    rollouts = []
    for b in range(head_avg.shape[1]):
        rollout = np.eye(head_avg.shape[-1])
        for l in range(head_avg.shape[0]):
            A = head_avg[l, b]
            A_aug = alpha * A + (1 - alpha) * np.eye(A.shape[0])
            A_aug = A_aug / A_aug.sum(-1, keepdims=True)
            rollout = A_aug @ rollout
        rollouts.append(rollout)

    rollouts = np.stack(rollouts, axis=0)  
    return rollouts # [jet, seq_len, seq_len]

def gensave_rollouts(model, num_events=10_000, alpha=0.8):
    """ saved as: <model_dir>/tests/<global_step>_<global_epoch>/rollouts_<alpha>_<num_events>.npz
    labels: rollouts [jet, const, const] (with cls_attn=rollouts[:, 0, :])
            jets [jet, const, feature]
            paddings [jet, const]
            labels [jet]
    """
    alpha = np.atleast_1d(alpha)
    attn, jets, paddings, labels = get_attn(model, num_events)
    aug_attn = augment_attn(attn, bb_layers=8)
    
    for a in alpha:
        rollouts = get_attn_rollouts(aug_attn, paddings, a)
        
        m = torch.load(model, "cpu")
        filename = m.dir / "tests" / f"{m.global_step}_{m.global_epoch}" / f"rollouts_{a}_{num_events}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, rollouts=rollouts, jets=jets, paddings=paddings, labels=labels)
        print(f"Rollouts saved as: '{filename}.npz'")
    
    return f"{filename}.npz"