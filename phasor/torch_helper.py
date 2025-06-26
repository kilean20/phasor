import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import namedtuple  # Import namedtuple
import matplotlib.pyplot as plt
from cma import CMAEvolutionStrategy
from copy import deepcopy

# Result = namedtuple('Result', ['x', 'fun', 'losses', 'weighted_losses', 'history','success'])
_div_factor = 5

class Result:
    def __init__(self, x, fun, losses, weighted_losses, history, success):
        self.x = x
        self.fun = fun
        self.losses = losses
        self.weighted_losses = weighted_losses
        self.history = history
        self.success = success

    def __repr__(self):
        return (f"Result(x={self.x}, fun={self.fun}, losses={self.losses}, "
                f"weighted_losses={self.weighted_losses}, history={self.history}, "
                f"success={self.success})")


class Standardizer:
    def __init__(self, info: dict, device: str = "cpu"):
        """
        Parameters:
        - info: dict mapping keys to either {'mean', 'std'} or {'min', 'max'}.
        - device: device to store all constants ('cpu', 'cuda', etc.)
        """
        self.device = torch.device(device)
        self.transforms = {}

        for key, cfg in info.items():
            entry = {}
            if 'mean' in cfg and 'std' in cfg and cfg['mean'] is not None and cfg['std'] is not None:
                entry['type'] = 'standard'
                entry['mean'] = torch.tensor(cfg['mean'], dtype=torch.float64, device=self.device)
                entry['std'] = torch.tensor(cfg['std'], dtype=torch.float64, device=self.device)
            elif 'min' in cfg and 'max' in cfg and cfg['min'] is not None and cfg['max'] is not None:
                entry['type'] = 'normalize'
                entry['min'] = torch.tensor(cfg['min'], dtype=torch.float64, device=self.device)
                entry['max'] = torch.tensor(cfg['max'], dtype=torch.float64, device=self.device)
                entry['range'] = entry['max'] - entry['min']
            else:
                raise ValueError(f"Key '{key}': Must provide either ('mean', 'std') or ('min', 'max').")
            self.transforms[key] = entry

    def transform(self, **kwargs):
        """
        Forward transform: apply standardization or normalization.
        """
        return {
            k: self._apply(v, self.transforms[k], forward=True) if k in self.transforms else v
            for k, v in kwargs.items()
        }

    def inverse(self, **kwargs):
        """
        Inverse transform: recover original scale.
        """
        return {
            k: self._apply(v, self.transforms[k], forward=False) if k in self.transforms else v
            for k, v in kwargs.items()
        }

    def _apply(self, x: torch.Tensor, cfg: dict, forward: bool):
        if cfg['type'] == 'standard':
            return (x - cfg['mean']) / cfg['std'] if forward else x * cfg['std'] + cfg['mean']
        elif cfg['type'] == 'normalize':
            return (x - cfg['min']) / cfg['range'] if forward else x * cfg['range'] + cfg['min']



def run_torch_optimizer_multi_kwargs(
    loss_func: callable, 
    x0_dict: Dict[str, torch.Tensor],
    lr: float = 0.01,
    optimizer_cls  = torch.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    max_iter: int = 300, 
    loss_weights: Optional[Dict[str, float]] = None, 
    low_fidelity_loss_func: Optional[callable] = None,
    patience: Optional[int] = None, 
    history_window: Optional[int] = None,  
    return_history: bool = False,
    plot_history: bool = False,
    ) -> Result:
    """
    Runs a Torch optimizer to minimize a given loss function.
    
    Arguments:
        loss_func: callable taking **kwargs and returning dict of losses.
        x0_dict: dict of initial parameters (torch.Tensor) to optimize.
        Others: same as run_torch_optimizer_multi_args
    """
    patience = patience if patience is not None else max(1, int(0.1 * max_iter))
    history_window = history_window if history_window is not None else max(1, int(0.1 * max_iter))
    if plot_history:
        return_history = True
    
    # Set requires_grad
    for param in x0_dict.values():
        param.requires_grad = True

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer_kwargs['lr'] = lr

    # Optimizer on the list of parameters
    optimizer = optimizer_cls(list(x0_dict.values()), **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=lr,
                                                    div_factor=_div_factor,
                                                    pct_start=0.1, 
                                                    final_div_factor=_div_factor,
                                                    epochs=max_iter, steps_per_epoch=1)
    
    iter_num = 0
    optimizer.zero_grad()
    losses = loss_func(**x0_dict)

    if loss_weights is None:
        loss_weights = {}
    loss_weights = {k: loss_weights.get(k, 1.0) for k in losses.keys()}
    
    weighted_losses = torch.stack([loss_weights[k] * v for k, v in losses.items() if v is not None])
    batched_loss = torch.sum(weighted_losses,dim=0)
    reg_loss = (batched_loss - weighted_losses[0]).max()
    loss = batched_loss.mean()

    best_losses = batched_loss.detach().clone()
    best_loss = loss.item()
    best_sol = {k: v.detach().clone() for k,v in x0_dict.items()}
    patience_counter = 0
    loss_history = [best_loss]

    if return_history:
        hist = {k: [v.detach().cpu().numpy()] for k, v in losses.items() if v is not None}
    else:
        hist = None

    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e4:
        print(f"Exiting early at iteration {iter_num + 1} due to large loss")
        return Result(x=best_sol, fun=best_losses, 
                      losses=losses, weighted_losses=weighted_losses, 
                      history=hist, success=False)

    loss.backward()
    optimizer.step()
    scheduler.step()

    for iter_num in range(1, max_iter):
        optimizer.zero_grad()

        if low_fidelity_loss_func and reg_loss <= 1e-6 and iter_num % 2 == 0:
            losses = low_fidelity_loss_func(**x0_dict)
        else:
            losses = loss_func(**x0_dict)

        weighted_losses = torch.stack([loss_weights[k] * v for k, v in losses.items() if v is not None])
        batched_loss = torch.sum(weighted_losses,dim=0)
        reg_loss = (batched_loss - weighted_losses[0]).max()
        loss = batched_loss.mean()
        
        if torch.isnan(loss) or torch.isinf(loss) or reg_loss.item() > 1e2:
            print(f"Exiting early at iteration {iter_num + 1} due to large loss")
            return Result(x=best_sol, fun=best_losses, 
                          losses=losses, weighted_losses=weighted_losses, 
                          history=hist, success=False)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if return_history:
            for k, v in losses.items():
                if v is not None:
                    hist[k].append(v.detach().cpu().numpy())

        if loss.item() < best_loss:
            best_losses = batched_loss.detach().clone()
            best_loss = loss.item()
            best_sol = {k: v.detach().clone() for k,v in x0_dict.items()}

        if len(loss_history) > 0.2 * max_iter + 2 * history_window:
            var_prev = np.var(loss_history[-2 * history_window: -history_window])
            var_now = np.var(loss_history[-history_window:])

            if var_now < 0.1 * var_prev:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iter_num + 1} due to loss stabilization.")
                    break

    if plot_history:
        plot_loss_history(hist)
    
    return Result(x=best_sol, fun=best_losses, 
                  losses=losses, weighted_losses=weighted_losses, 
                  history=hist, success=True)
        
def run_torch_LBFGS_multi_kwargs(
    loss_func: callable, 
    x0_dict: Dict[str, torch.Tensor],
    lr: float = 1.0,
    optimizer_cls = torch.optim.LBFGS,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    max_iter: int = 300, 
    loss_weights: Optional[Dict[str, float]] = None, 
    low_fidelity_loss_func: Optional[callable] = None,
    patience: Optional[int] = None, 
    history_window: Optional[int] = None,  
    return_history: bool = False,
    plot_history: bool = False,
) -> Result:

    patience = patience if patience is not None else max(1, int(0.1 * max_iter))
    history_window = history_window if history_window is not None else max(1, int(0.1 * max_iter))
    if plot_history:
        return_history = True

    for param in x0_dict.values():
        param.requires_grad = True

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer_kwargs['lr'] = lr

    optimizer = optimizer_cls(list(x0_dict.values()), **optimizer_kwargs)

    if loss_weights is None:
        loss_weights = {}
    loss_weights = {k: loss_weights.get(k, 1.0) for k in loss_func(**x0_dict).keys()}

    best_loss = float('inf')
    best_losses = None
    best_sol = {k: v.detach().clone() for k, v in x0_dict.items()}
    patience_counter = 0
    loss_history = []

    if return_history:
        hist = {k: [] for k in loss_weights}
    else:
        hist = None

    def closure():
        optimizer.zero_grad()
        losses = loss_func(**x0_dict)
        weighted_losses = torch.stack([loss_weights[k] * v for k, v in losses.items() if v is not None])
        batched_loss = torch.sum(weighted_losses, dim=0)
        loss = batched_loss.mean()
        loss.backward()
        return loss

    for iter_num in range(max_iter):
        if low_fidelity_loss_func and iter_num % 2 == 0:
            current_losses = low_fidelity_loss_func(**x0_dict)
        else:
            current_losses = loss_func(**x0_dict)

        weighted_losses = torch.stack([loss_weights[k] * v for k, v in current_losses.items() if v is not None])
        batched_loss = torch.sum(weighted_losses, dim=0)
        loss = batched_loss.mean()

        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e4:
            print(f"Exiting early at iteration {iter_num + 1} due to large or invalid loss")
            return Result(x=best_sol, fun=best_losses, losses=current_losses,
                          weighted_losses=weighted_losses, history=hist, success=False)

        optimizer.step(closure)
        loss_history.append(loss.item())

        if return_history:
            for k, v in current_losses.items():
                if v is not None:
                    hist[k].append(v.detach().cpu().numpy())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_losses = batched_loss.detach().clone()
            best_sol = {k: v.detach().clone() for k, v in x0_dict.items()}

        if len(loss_history) > 2 * history_window:
            var_prev = np.var(loss_history[-2 * history_window: -history_window])
            var_now = np.var(loss_history[-history_window:])
            if var_now < 0.1 * var_prev:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iter_num + 1} due to loss stabilization.")
                    break

    if plot_history:
        plot_loss_history(hist)

    return Result(x=best_sol, fun=best_losses,
                  losses=current_losses, weighted_losses=weighted_losses,
                  history=hist, success=True)

def run_torch_optimizer_multi_args(
    loss_func: callable, 
    l_x0: Tuple[torch.Tensor],
    lr: float = 0.01,
    optimizer_cls  = torch.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    max_iter: int = 300, 
    loss_weights: Optional[Dict[str, float]] = None, 
    low_fidelity_loss_func: Optional[callable] = None,
    patience: Optional[int] = None, 
    history_window: Optional[int] = None,  
    return_history: bool = False,
    plot_history: bool = False,
    ) -> Result:
    """
    Runs a Torch optimizer to minimize a given loss function.

    Args:
        loss_func (callable): A function that computes various loss values as a dictionary.
            The first key should be the main loss (e.g., 'train_loss'), and the remaining keys 
            should correspond to regularization losses (e.g., 'reg_loss1', 'reg_loss2', etc.).
        l_x0 Tuple of torch.Tensor: Initial parameters (tensor) to optimize.
        max_iter (int, optional): Maximum number of optimization iterations. Default is 100.
        loss_weights (Dict[str, float], optional): Weights for the individual losses, where keys
            correspond to the keys in the loss dictionary. Default is None.
        low_fidelity_loss_func (callable, optional): Alternative loss function for lower fidelity computations. 
            Default is None.
        patience (int, optional): Number of iterations with no improvement before stopping. Default is 
            5% of max_iter.
        history_window (int, optional): Number of iterations to consider for calculating variance in loss. 
            Default is 5% of max_iter.
        return_history (bool, optional): Whether to return the loss history. Default is False.

    Returns:
        namedtuple: 
            - 'x': Best parameter solution (tensor).
            - 'fun': Best loss value (tensor).
            - 'history': History of losses for each parameter (dictionary).
    """

    # Initialize parameters
    patience = patience if patience is not None else max(1, int(0.1 * max_iter))
    history_window = history_window if history_window is not None else max(1, int(0.1 * max_iter))
    if plot_history:
        return_history = True
    
    for arg in l_x0:
        arg.requires_grad = True

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer_kwargs['lr'] = lr
    optimizer = optimizer_cls(l_x0, **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=lr,
                                                    div_factor=_div_factor,
                                                    pct_start=0.1, 
                                                    final_div_factor=_div_factor,
                                                    epochs=max_iter, steps_per_epoch=1)
    

    # 0th iteration
    iter_num = 0
    optimizer.zero_grad()
    losses = loss_func(*l_x0)


    if loss_weights is None:
        loss_weights = {}
    loss_weights = {k: loss_weights.get(k, 1.0) for k in losses.keys()}
    
    weighted_losses = torch.stack([loss_weights[k] * v for k, v in losses.items() if v is not None])
    batched_loss = torch.sum(weighted_losses,dim=0)
    reg_loss = (batched_loss - weighted_losses[0]).max()
    loss = batched_loss.mean()

    best_losses = batched_loss.detach().clone()
    best_loss = loss.item()
    best_sol = [x0.detach().clone() for x0 in l_x0]
    patience_counter = 0
    loss_history = [best_loss]
    
    if return_history:
        hist = {k: [v.detach().cpu().numpy()] for k, v in losses.items()  if v is not None}
    else:
        hist = None

    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e4:
        print(f"Exiting early at iteration {iter_num + 1} due to large loss")
        return Result(x=best_sol, fun=best_losses, 
                      losses=losses, weighted_losses=weighted_losses, 
                      history=hist, success=False)

    loss.backward()
    optimizer.step()
    scheduler.step()


    for iter_num in range(1, max_iter):
        optimizer.zero_grad()

        if low_fidelity_loss_func and reg_loss <= 1e-6 and iter_num % 2 == 0:
            losses = low_fidelity_loss_func(*l_x0)
        else:
            losses = loss_func(*l_x0)

        weighted_losses = torch.stack([loss_weights[k] * v for k, v in losses.items() if v is not None])
        batched_loss = torch.sum(weighted_losses,dim=0)
        reg_loss = (batched_loss - weighted_losses[0]).max()
        loss = batched_loss.mean()
        
        if torch.isnan(loss) or torch.isinf(loss) or reg_loss.item() > 1e2:
            print(f"Exiting early at iteration {iter_num + 1} due to large loss")
            return Result(x=best_sol, fun=best_losses, 
                          losses=losses, weighted_losses=weighted_losses, 
                          history=hist, success=False)
        
        # if loss.item() > 1e1:
            # print("loss too large. gradient clipped")
            # torch.nn.utils.clip_grad_norm_(x0, max_norm=1.0)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if return_history:
            for k, v in losses.items():
                if v is not None:
                    hist[k].append(v.detach().cpu().numpy())

        # Update best solution if current loss is lower
        if loss.item() < best_loss:
            best_losses = batched_loss.detach().clone()
            best_loss = loss.item()
            best_sol = [x0.detach().clone() for x0 in l_x0]

        # Early stopping based on dynamic variance in loss
        if len(loss_history) > 0.2 * max_iter + 2 * history_window:
            var_prev = np.var(loss_history[-2 * history_window: -history_window])
            var_now = np.var(loss_history[-history_window:])

            if var_now < 0.1 * var_prev:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iter_num + 1} due to loss stabilization.")
                    break

    # Define and return the named tuple
    if plot_history:
        plot_loss_history(hist)
    
    return Result(x=best_sol, fun=best_losses, 
                  losses=losses, weighted_losses=weighted_losses, 
                  history=hist, success=True)
    


def run_torch_optimizer(
    loss_func: callable, 
    x0: torch.Tensor,
    lr: float = 0.01,
    optimizer_cls  = torch.optim.Adam,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    max_iter: int = 300, 
    loss_weights: Optional[Dict[str, float]] = None, 
    low_fidelity_loss_func: Optional[callable] = None,
    patience: Optional[int] = None, 
    history_window: Optional[int] = None,  
    return_history: bool = False,
    plot_history: bool = False,
) -> Result:
    """
    Runs a Torch optimizer to minimize a given loss function.

    Args:
        loss_func (callable): A function that computes various loss values as a dictionary.
            The first key should be the main loss (e.g., 'train_loss'), and the remaining keys 
            should correspond to regularization losses (e.g., 'reg_loss1', 'reg_loss2', etc.).
        x0 (torch.Tensor): Initial parameters (tensor) to optimize.
        max_iter (int, optional): Maximum number of optimization iterations. Default is 100.
        loss_weights (Dict[str, float], optional): Weights for the individual losses, where keys
            correspond to the keys in the loss dictionary. Default is None.
        low_fidelity_loss_func (callable, optional): Alternative loss function for lower fidelity computations. 
            Default is None.
        patience (int, optional): Number of iterations with no improvement before stopping. Default is 
            5% of max_iter.
        history_window (int, optional): Number of iterations to consider for calculating variance in loss. 
            Default is 5% of max_iter.
        return_history (bool, optional): Whether to return the loss history. Default is False.

    Returns:
        namedtuple: 
            - 'x': Best parameter solution (tensor).
            - 'fun': Best loss value (tensor).
            - 'history': History of losses for each parameter (dictionary).
    """

    # Initialize parameters
    patience = patience if patience is not None else max(1, int(0.1 * max_iter))
    history_window = history_window if history_window is not None else max(1, int(0.1 * max_iter))
    if plot_history:
        return_history = True
    
    x0.requires_grad = True
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer_kwargs['lr'] = lr
    optimizer = optimizer_cls([x0], **optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=lr,
                                                    div_factor=_div_factor,
                                                    pct_start=0.1, 
                                                    final_div_factor=_div_factor,
                                                    epochs=max_iter, steps_per_epoch=1)
    

    # 0th iteration
    iter_num = 0
    optimizer.zero_grad()
    losses = loss_func(x0)


    if loss_weights is None:
        loss_weights = {}
    loss_weights = {k: loss_weights.get(k, 1.0) for k in losses.keys()}
    
    weighted_losses = torch.stack([loss_weights[k] * v for k, v in losses.items() if v is not None])
    batched_loss = torch.sum(weighted_losses,dim=0)
    reg_loss = (batched_loss - weighted_losses[0]).max()
    loss = batched_loss.mean()

    best_losses = batched_loss.detach().clone()
    best_loss = loss.item()
    best_sol = x0.detach().clone()
    patience_counter = 0
    loss_history = [best_loss]
    
    if return_history:
        hist = {k: [v.detach().cpu().numpy()] for k, v in losses.items()  if v is not None}
    else:
        hist = None

    if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e2:
        print(f"Exiting early at iteration {iter_num + 1} due to large loss")
        return Result(x=best_sol, fun=best_losses, 
                      losses=losses, weighted_losses=weighted_losses, 
                      history=hist, success=False)

    loss.backward()
    optimizer.step()
    scheduler.step()


    for iter_num in range(1, max_iter):
        optimizer.zero_grad()

        if low_fidelity_loss_func and reg_loss <= 1e-6 and iter_num % 2 == 0:
            losses = low_fidelity_loss_func(x0)
        else:
            losses = loss_func(x0)

        weighted_losses = torch.stack([loss_weights[k] * v for k, v in losses.items() if v is not None])
        batched_loss = torch.sum(weighted_losses,dim=0)
        reg_loss = (batched_loss - weighted_losses[0]).max()
        loss = batched_loss.mean()
        
        if torch.isnan(loss) or torch.isinf(loss) or reg_loss.item() > 1e2:
            print(f"Exiting early at iteration {iter_num + 1} due to large loss")
            return Result(x=best_sol, fun=best_losses, 
                          losses=losses, weighted_losses=weighted_losses, 
                          history=hist, success=False)
        
        # if loss.item() > 1e1:
            # print("loss too large. gradient clipped")
            # torch.nn.utils.clip_grad_norm_(x0, max_norm=1.0)

        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if return_history:
            for k, v in losses.items():
                if v is not None:
                    hist[k].append(v.detach().cpu().numpy())

        # Update best solution if current loss is lower
        if loss.item() < best_loss:
            best_losses = batched_loss.detach().clone()
            best_loss = loss.item()
            best_sol = x0.detach().clone()

        # Early stopping based on dynamic variance in loss
        if len(loss_history) > 0.2 * max_iter + 2 * history_window:
            var_prev = np.var(loss_history[-2 * history_window: -history_window])
            var_now = np.var(loss_history[-history_window:])

            if var_now < 0.05 * var_prev:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iter_num + 1} due to loss stabilization.")
                    break

    # Define and return the named tuple
    if plot_history:
        plot_loss_history(hist)
    
    return Result(x=best_sol, fun=best_losses, 
                  losses=losses, weighted_losses=weighted_losses, 
                  history=hist, success=True)
    


def plot_loss_history(history: Dict[str, np.ndarray],return_fig=False):
    """
    Plots the loss history for each loss type over the epochs.
    Args:
        history (Dict[str, List[np.ndarray]]): A dictionary where the keys are names of loss terms (e.g., 
            'train_loss', 'reg_loss') and the values are lists of numpy arrays. Each numpy array represents
            the loss values recorded over epochs for a specific run (e.g., across batches or iterations).
        return_fig (bool): If True, returns the figure object for further customization.
    """
    k = list(history.keys())[0]
    v = np.array(history[k])
    if len(v)<1:
        if return_fig:
            return None, None
        else:
            return
    if v.ndim > 1:
        return plot_batched_loss_history(history,return_fig=return_fig)
        
    fig, ax = plt.subplots(figsize=(4, 3))  # Two subplots in one row

    c = 0  # Color counter for each loss type
    for k, v in history.items():
        v = np.array(v)  # Convert to numpy array for ease of processing

        # Skip this loss type if all values are NaN
        if np.isnan(v).all():
            continue

        # Plot the mean loss across runs (axis 0 is usually epoch)
        ax.plot(v, color=f'C{c}', label=k)
        c += 1  # Increment color index for the next loss type

    # Labeling the axes for both subplots
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.legend(loc='best')  # Add legend to the first plot (mean losses)
    fig.tight_layout()  # Adjust layout so the plots fit nicely
    
    if return_fig:
        return fig, ax
    else:
        plt.show()


def plot_batched_loss_history(history: Dict[str, np.ndarray],return_fig=False):
    """
    Plots the loss history for each loss type over the epochs.
    Args:
        history (Dict[str, List[np.ndarray]]): A dictionary where the keys are names of loss terms (e.g., 
            'train_loss', 'reg_loss') and the values are lists of numpy arrays. Each numpy array represents
            the loss values recorded over epochs for a specific run (e.g., across batches or iterations).
        return_fig (bool): If True, returns the figure object for further customization.
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))  # Two subplots in one row

    c = 0  # Color counter for each loss type
    for k, v in history.items():
        v = np.array(v)  # Convert to numpy array for ease of processing

        # Skip this loss type if all values are NaN
        if np.isnan(v).all():
            continue

        # Plot the mean loss across runs (axis 0 is usually epoch)
        if v.ndim>1:
            ax[0].plot(v.mean(axis=1), color=f'C{c}', label=k)
        
        # Plot individual loss trajectories for each run
        for vcol in v.T:
            ax[1].plot(vcol, color=f'C{c}', alpha=0.3, lw=1)  # Lightened lines for individual losses
        
        c += 1  # Increment color index for the next loss type

    # Labeling the axes for both subplots
    ax[0].set_xlabel('epochs')
    ax[1].set_xlabel('epochs')
    ax[0].set_ylabel('loss')

    ax[0].legend(loc='best')  # Add legend to the first plot (mean losses)

    fig.tight_layout()  # Adjust layout so the plots fit nicely
    
    if return_fig:
        return fig, ax
    else:
        plt.show()
