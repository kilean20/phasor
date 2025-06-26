from typing import List, Optional, Dict
import pickle
import numpy as np
import psutil
import time
import datetime
from copy import deepcopy as copy

import torch
_dtype = torch.float32
svmem = psutil.virtual_memory()

def get_mem_usage():
    svmem = psutil.virtual_memory()
    return svmem.used/svmem.available


class dummy_scheduler:
    def __init__(self,lr):
        self.lr = lr
    def step(self,*args,**kwargs):
        pass
    def get_last_lr(self,*args,**kwargs):
        return [self.lr]

    
def dropout_scheduler(epoch,epochs):
     return 0.5*np.exp(- 9*(epoch - 0.3*epochs)**2/epochs**2 )
    
    
class scalar:
    """
    A class for normalizing and unnormalizing scalar values.

    Args:
        **kwargs: Keyword arguments for initialization. 
                  Should include either 'xmin', 'ymin', 'xmax' (or 'xdiff'), 'ymax' (or 'ydiff'), 
                                     or 'xstd', 'xmean', 'ystd', 'ymean'.
                  Optionally, include 'fname' for file operations.

    Example:
        ```python
        # Create an instance of ScalarConstructor
        scaler = ScalarConstructor(xmin=0, xmax=100, ymin=0, ymax=1, fname='scaler.pkl')

        # Normalize and unnormalize values
        x_normalized = scaler.normalize_x(x)
        x_unnormalized = scaler.unnormalize_x(x_normalized)
        ```

    Attributes:
        mode (str): Either 'minmax' or 'standard' depending on the normalization mode.
        info (dict): Dictionary containing normalization information.
    """
    def __init__(self,**kwargs):
        info = kwargs.copy()
        self.info = info
        for k, v in info.items():
            if not isinstance(v, str):
                v = np.array(v)
                info[k] = v.reshape(-1, v.size)
                
        if 'fname' in info:
            fname = info['fname']
            if fname[-4:]!='.pkl':
                fname = fname+'.pkl'
            info['fname'] = fname
                
        if 'xmin' in info and 'ymin' in info:
            self.mode = 'minmax'
            if 'xmax' in info:
                info['xdiff']=info['xmax']-info['xmin']
            else:
                assert 'xdiff' in info
            if 'ymax' in info:
                info['ydiff']=info['ymax']-info['ymin']
            else:
                assert 'ydiff' in info  
            if 'fname' in info:
                self._dump(info)
                
        elif 'xstd' in info and 'xmean' in info and 'ystd' in info and 'ymean' in info:
            self.mode = 'standard'
            if 'fname' in info:
                self._dump(info)
            
        elif 'fname' in info:
            self._load(info['fname'])
            if 'xdiff' in self.info and 'ydiff' in self.info:
                self.mode = 'minmax'
            elif 'xstd' in self.info and 'ystd' in self.info:
                self.mode = 'standard'
            else:
                raise ValueError('not valid file')
        else:
            raise ValueError('not valid Inputs')
            
            
            
    def _dump(self,info):
        pickle.dump(info,open(info['fname'],'wb'))
        
    def _load(self,fname):
        self.info = pickle.load(open(fname,'rb'))
        if 'xdiff' in self.info:
            self.info['xdiff'] = torch.tensor(self.info['xdiff'],dtype=_dtype)
        if 'ydiff' in self.info:
            self.info['ydiff'] = torch.tensor(self.info['ydiff'],dtype=_dtype)
        if 'xmin' in self.info:
            self.info['xmin' ] = torch.tensor(self.info['xmin' ],dtype=_dtype)
        if 'ymin' in self.info:
            self.info['ymin' ] = torch.tensor(self.info['ymin' ],dtype=_dtype)  
        if 'xstd' in self.info:
            self.info['xstd' ] = torch.tensor(self.info['xstd' ],dtype=_dtype)
        if 'xmean' in self.info:
            self.info['xmean'] = torch.tensor(self.info['xmean'],dtype=_dtype)
        if 'ystd' in self.info:
            self.info['ystd' ] = torch.tensor(self.info['ystd' ],dtype=_dtype)
        if 'ymean' in self.info:
            self.info['ymean'] = torch.tensor(self.info['ymean'],dtype=_dtype)
        
    def normalize_x(self,x):
        if self.mode == 'minmax':
            return (x-self.info['xmin'])/self.info['xdiff']
        else:
            return (x-self.info['xmean'])/self.info['xstd']
        
    def unnormalize_x(self,x):
        if self.mode == 'minmax':
            return x*self.info['xdiff']+self.info['xmin']
        else:
            return x*self.info['xstd']+self.info['xmean']

    def normalize_y(self,y):
        if self.mode == 'minmax':
            return (y-self.info['ymin'])/self.info['ydiff']
        else:
            return (y-self.info['ymean'])/self.info['ystd']
        
    def unnormalize_y(self,y):
        if self.mode == 'minmax':
            return y*self.info['ydiff']+self.info['ymin']
        else:
            return y*self.info['ystd']+self.info['ymean']
        
        
        
def construct_model( 
    input_dim: int, 
    output_dim: int, 
    input_phase_feature_dim: int = 0,
    linear_nodes: [List[int],int] = None,
    hidden_nodes: [List[int],int] = None,
    activation: torch.nn.Module = torch.nn.ELU(),
    return_model_info: bool = True,
    ):
    if isinstance(linear_nodes,int):
        linear_nodes = [linear_nodes]
    if linear_nodes is None:
        n1 = 2**min(input_dim+6, 2048)
        linear_nodes = [int(n1/(i+1)) for i in range(1)]
        
    if isinstance(hidden_nodes,int):
        hidden_nodes = [hidden_nodes]
    if hidden_nodes is None:
        n1 = 2**min(input_dim+5, 2048)
        hidden_nodes = [int(n1/(i+1)) for i in range(5)]
        
    if isinstance(activation,str):
        activation = getattr(torch.nn, activation)()
        
    if input_phase_feature_dim > 0:
        model = _ModelWithPhaseInput(
            input_dim = input_dim,
            output_dim = output_dim,
            input_phase_feature_dim = input_phase_feature_dim,
            linear_nodes = linear_nodes,
            hidden_nodes = hidden_nodes,
            activation = activation,
            )
    else:
        model = _Model(
            input_dim = input_dim,
            output_dim = output_dim,
            linear_nodes = linear_nodes,
            hidden_nodes = hidden_nodes,
            activation = activation,
            )
        
    
    if return_model_info:
        if activation:
            activation = activation.__class__.__name__
        return model, {'input_dim':input_dim,
                       'output_dim': output_dim,
                       'input_phase_feature_dim': input_phase_feature_dim,
                       'linear_nodes': linear_nodes,
                       'hidden_nodes': hidden_nodes,
                       'activation': activation,
                      }
    else:
        return model, None


    
class _Model(torch.nn.Module):
    """
    A custom neural network model with a nonlinear residual block, a linear base layer, and optional dropout that can be used to escape local minima during model training.

    Args:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        hidden_nodes (List[int]): A list of integers specifying the number of hidden nodes in each layer of the nonlinear residual block.
        linear_nodes (int, optional): The number of hidden nodes in the linear base layer. Defaults to 256.
        activation (torch.nn.Module, optional): The activation function to use after each layer in the nonlinear residual block. Defaults to torch.nn.ELU().

    Example:
        ```python
        # Create a new instance of the model
        model = ModelWithDropout(input_dim=10, output_dim=5, hidden_nodes=[16, 32])

        # Train the model

        # Save the model state
        state_dict = model.state_dict()

        # Create a new instance and load the state
        loaded_model = ModelWithDropout(state_dict=state_dict)

        # Use the loaded model
        y_hat = loaded_model(x)
        ```

    Attributes:
        activation (torch.nn.Module): The activation function used in the nonlinear residual block.
        nonlinear_residual (torch.nn.Sequential): The nonlinear residual block.
        linear_base (torch.nn.Sequential): The linear base layer.
        dropout_p (float): The probability of dropout. Set to 0.0 by default.
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 linear_nodes: List[int] = None,
                 hidden_nodes: List[int] = None,
                 activation: torch.nn.Module = torch.nn.ELU(),
                 ):
        super().__init__()
        
        # linear base
        if isinstance(linear_nodes,int):
            linear_nodes = [linear_nodes]
        if linear_nodes is None:
            n1 = 2**min(input_dim+6, 2048)
            linear_nodes = [int(n1/(i+1)) for i in range(1)]
        layers = [torch.nn.Linear(input_dim, linear_nodes[0])]
        if len(linear_nodes) > 1:
            for i in range(len(linear_nodes) - 1):
                layers.append(torch.nn.Linear(linear_nodes[i], linear_nodes[i + 1]))
        layers.append(torch.nn.Linear(linear_nodes[-1], output_dim))
        self.linear_base = torch.nn.Sequential(*layers)

        # nonlinear residual
        self.activation = activation
        if isinstance(hidden_nodes,int):
            hidden_nodes = [hidden_nodes]
        if hidden_nodes is None:
            n1 = 2**min(input_dim+5, 2048)
            hidden_nodes = [int(n1/(i+1)) for i in range(5)]
        layers = [torch.nn.Linear(input_dim, hidden_nodes[0])]
        if self.activation:
            layers.append(self.activation)
        if len(hidden_nodes) > 1:
            for i in range(len(hidden_nodes) - 1):
                layers.append(torch.nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
                if self.activation:
                    layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_nodes[-1], output_dim))
        self.nonlinear_residual = torch.nn.Sequential(*layers)

        # dropout rate is 0 by default. Manually adjust it during training as needed. 
        self.dropout_p = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        z_linear = self.linear_base(x)
        if self.dropout_p > 0:
            z_linear = torch.nn.functional.dropout(z_linear, p=self.dropout_p, training=self.training)
        z_nonlinear = self.nonlinear_residual(x)
        return z_linear + z_nonlinear


class _ModelWithPhaseInput(torch.nn.Module):
    """
    A custom neural network model with a nonlinear residual block, a linear base layer, and optional dropout that can be used to escape local minima during model training.

    Args:
        input_dim (int): The dimension of the input data.
        output_dim (int): The dimension of the output data.
        input_phase_feature_dim (int, optional): The dimension of the phase feature in the input data.
                                                 * the first input_phase_feature_dim in input variable must be phases
        linear_nodes (List[int], optional): A list of integers specifying the number of hidden nodes in each layer of the linear base layer. If not provided, it is computed based on input_dim. Defaults to None.
        hidden_nodes (List[int], optional): A list of integers specifying the number of hidden nodes in each layer of the nonlinear residual block. If not provided, it is computed based on input_dim. Defaults to None.
        activation (torch.nn.Module, optional): The activation function to use after each layer in the nonlinear residual block. Defaults to torch.nn.ELU().

    Example:
        ```python
        # Create a new instance of the model
        model = _ModelWithPhaseInput(input_dim=10, output_dim=5, input_phase_feature_dim=2, hidden_nodes=[16, 32])

        # Train the model

        # Save the model state
        state_dict = model.state_dict()

        # Create a new instance and load the state
        loaded_model = _ModelWithPhaseInput(state_dict=state_dict)

        # Use the loaded model
        y_hat = loaded_model(x)
        ```

    Attributes:
        activation (torch.nn.Module): The activation function used in the nonlinear residual block.
        nonlinear_residual (torch.nn.Sequential): The nonlinear residual block.
        linear_base (torch.nn.Sequential): The linear base layer.
        dropout_p (float): The probability of dropout. Set to 0.0 by default.
    """

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 input_phase_feature_dim: int,
                 linear_nodes: List[int] = None,
                 hidden_nodes: List[int] = None,
                 activation: torch.nn.Module = torch.nn.ELU(),
                 ):
        super().__init__()
        
        input_dim = input_dim  + input_phase_feature_dim  # x[1, ph1] ->  [x1, sin(ph1), cos(ph1)]
        self.n_input_phase_feature = input_phase_feature_dim
        
        # linear base
        if isinstance(linear_nodes,int):
            linear_nodes = [linear_nodes]
        if linear_nodes is None:
            n1 = 2**min(input_dim+6, 2048)
            linear_nodes = [int(n1/(i+1)) for i in range(1)]
            
        layers = [torch.nn.Linear(input_dim, linear_nodes[0])]
        if len(linear_nodes) > 1:
            for i in range(len(linear_nodes) - 1):
                layers.append(torch.nn.Linear(linear_nodes[i], linear_nodes[i + 1]))
        layers.append(torch.nn.Linear(linear_nodes[-1], output_dim))
        self.linear_base = torch.nn.Sequential(*layers)

        # nonlinear residual
        self.activation = activation
        if isinstance(hidden_nodes,int):
            hidden_nodes = [hidden_nodes]
        if hidden_nodes is None:
            n1 = 2**min(input_dim+5, 2048)
            hidden_nodes = [int(n1/(i+1)) for i in range(5)]
        layers = [torch.nn.Linear(input_dim, hidden_nodes[0])]
        if self.activation:
            layers.append(self.activation)
        if len(hidden_nodes) > 1:
            for i in range(len(hidden_nodes) - 1):
                layers.append(torch.nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
                if self.activation:
                    layers.append(self.activation)
        layers.append(torch.nn.Linear(hidden_nodes[-1], output_dim))
        self.nonlinear_residual = torch.nn.Sequential(*layers)
        
        self.dropout_p = 0.0

    def forward(self, x):
        x_phase = x[:,:self.n_input_phase_feature]
        x_feature = x[:,self.n_input_phase_feature:]
        sin = torch.sin(x_phase)
        cos = torch.cos(x_phase)
        x_feature = torch.cat((sin,cos,x_feature),dim=-1)
        z_linear = self.linear_base(x_feature)
        if self.dropout_p > 0:
            z_linear = torch.nn.functional.dropout(z_linear, p=self.dropout_p, training=self.training)
        z_nonlinear = self.nonlinear_residual(x_feature)
        return z_linear + z_nonlinear

    
    
        
# class _model_with_phaseinput_and_dropout(torch.nn.Module):
#     def __init__(self,
#                  hidden_nodes,
#                  linear_nodes = 256,
#                  activation = torch.nn.ELU(),
#                 ):
#         '''
#         linear single layer net for linear x-y relation
#         nonlinear res block for addtional linear and nonlinear x-y relation
#         dropout can be used for ecaping from local minimum
#         '''
#         super().__init__()   
#         self.activation = activation or torch.nn.ELU()

#         # nonlinear residual
#         assert len(hidden_nodes)>2
#         layers = [torch.nn.Linear(4,hidden_nodes[0])]
#         if self.activation:
#             layers.append(self.activation)
#         for i in range(len(hidden_nodes)-1):
#             layers.append(torch.nn.Linear(hidden_nodes[i],hidden_nodes[i+1]))
#             if self.activation:
#                 layers.append(self.activation)
#         layers.append(torch.nn.Linear(hidden_nodes[i+1], 1))
#         self.nonlinear_residual = torch.nn.Sequential(*layers)

#         # linear base
#         layers = [torch.nn.Linear(4,linear_nodes),
#                   torch.nn.Linear(linear_nodes, 1)]
#         self.linear_base = torch.nn.Sequential(*layers)
#         self.dropout_p = 0.0
        
#     def forward(self, x):
#         x_feature = x[:,1:]
#         x_phase = x[:,:1]
#         sin = torch.sin(x_phase)
#         cos = torch.cos(x_phase)
#         x_feature = torch.cat((sin,cos,x_feature),dim=-1)
#         z_linear = self.linear_base(x_feature)
#         if self.dropout_p > 0:
#             z_linear = torch.nn.functional.dropout(z_linear, p=self.dropout_p, training=self.training)
#         z_nonlinear = self.nonlinear_residual(x_feature)
#         return z_linear + z_nonlinear
        
        
        
    
    
def train(
    model,x,y,epochs,lr,
    batch_size=None,
    shuffle=True,
    validation_split=0.0,
    criterion=torch.nn.MSELoss(),
    optimizer = torch.optim.Adam,
    optim_args = None,
    optimizer_state_dict = None,
    lr_scheduler = True,
    dropout_stabilization = False,
    prev_history = None,
    load_best = True,
    training_timeout = np.inf,
    verbose = False,
    fname_model = 'model.pt',
    fname_opt = 'opt.pt',
    fname_history = 'history.pkl',
    ):
    
    if isinstance(optimizer,str):
        optimizer = getattr(torch.optim, optimizer)
    if isinstance(criterion,str):
        criterion =  getattr(torch.nn, criterion)()

    if verbose:
        print("Train Function Arguments:",datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print(f"  - model: {model.__class__.__name__}")
        print(f"  - x: {x.shape if hasattr(x, 'shape') else type(x)}")
        print(f"  - y: {y.shape if hasattr(y, 'shape') else type(y)}")
        print(f"  - epochs: {epochs}")
        print(f"  - lr: {lr}")
        print(f"  - batch_size: {batch_size}")
        print(f"  - shuffle: {shuffle}")
        print(f"  - validation_split: {validation_split}")
        print(f"  - criterion: {criterion.__class__.__name__}")
        print(f"  - optimizer: {optimizer.__name__}")
        print(f"  - optim_args: {optim_args}")
        print(f"  - optimizer_state_dict: {optimizer_state_dict}")
        print(f"  - lr_scheduler: {lr_scheduler}")
        print(f"  - dropout_stabilization: {dropout_stabilization}")
        print(f"  - prev_history: {prev_history}")
        print(f"  - load_best: {load_best}")
        print(f"  - training_timeout: {training_timeout}")
        print(f"  - verbose: {verbose}")
        print(f"  - fname_model: {fname_model}")
        print(f"  - fname_opt: {fname_opt}")
        print(f"  - fname_history: {fname_history}")
        print()
    
    # get dtype and device of the input layer of the model
    n,d = y.shape
    if verbose:
        print("Model Paramers:")
    for name, p in model.named_parameters():
        if verbose:
            print(f"  - name: {name}, shape: {p.shape}, dtype: {p.dtype}, device: {p.device}")
        if len(p.shape) == 1:
            continue
        if 'linear_base' in name:
            device = p.device
            dtype = p.dtype
    if verbose:
        print()
 
          
    x=torch.tensor(x,dtype=dtype)
    y=torch.tensor(y,dtype=dtype)
    
    ntrain = len(x)
    assert len(x) == len(y)
    batch_size = batch_size or ntrain
    if validation_split>0.0:
        nval = int(validation_split*ntrain)
        ntrain = ntrain-nval
        val_x = x[:nval]
        val_y = y[:nval]
        x = x[nval:]
        y = y[nval:]
        nbatch_val = int(nval/batch_size)
        if nbatch_val==0:
            val_batch_size = nval
            nbatch_val = 1
        else:
            val_batch_size = batch_size
    else:
        nbatch_val=0
    train_batch_size = min(batch_size,ntrain)
    nbatch_train = int(ntrain/train_batch_size)
    if nbatch_train<16:
        nbatch_train = 1
        while(ntrain/nbatch_train > 16 or nbatch_train<16):
            nbatch_train *= 2            
        train_batch_size = int(ntrain/nbatch_train)

    training_timeout = training_timeout
    t0 = time.monotonic()
    assert epochs>0
    optim_args = optim_args or {}
    

    opt = optimizer(model.parameters(filter(lambda p: p.requires_grad, model.parameters())),lr=lr,**optim_args)
    if optimizer_state_dict is not None:
        opt.load_state_dict(optimizer_state_dict)
        

               
    if prev_history is None:
        history = {
            'train_loss':[],
            'val_loss'  :[],
            'lr'        :[],
            }
    else:
        assert "train_loss" in prev_history
        history = prev_history 
        if "lr" not in history:
            history['lr'] = [None]*len(history["train_loss"])
    epoch_start = len(history['train_loss'])
        
    if lr_scheduler:
        last_epoch = epoch_start*train_batch_size
        if last_epoch == 0:
            last_epoch = -1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, 
            max_lr=lr,
            div_factor=int(np.clip(epochs/500,a_min=1,a_max=20)),
            pct_start=0.05, 
            final_div_factor=int(np.clip(epochs/50,a_min=10,a_max=1e4)),
            epochs=epochs, steps_per_epoch=nbatch_train, last_epoch=last_epoch)
    else:       
        scheduler = dummy_scheduler(lr)
        
    best = np.inf
    model.train()
    epoch = epoch_start-1
    save_epoch = epoch
    
    
    if dropout_stabilization:
        assert hasattr(model,'dropout_p')
        
        
    if verbose:
        print("Training begin at: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        print()
    
    while(True):
        epoch += 1
        if epoch>=epoch_start + epochs:
            break
        lr_ = scheduler.get_last_lr()[0]
        history['lr'].append(lr_)
        if dropout_stabilization:
            model.dropout_p = dropout_scheduler(epoch,epochs)
        else:
            model.dropout_p = 0
        if shuffle:
            p = np.random.permutation(len(x))
            x = x[p]
            y = y[p]

        train_loss = 0
        for i in range(nbatch_train):
            i1 = i*train_batch_size
            i2 = i1+train_batch_size
            x_ = x[i1:i2,:]
            y_ = y[i1:i2,:]
            x_ = x_.to(device)
            y_ = y_.to(device)
            opt.zero_grad()
            y_pred_ = model(x_)
            loss = criterion(y_, y_pred_)
            loss.backward()
            opt.step()
            scheduler.step()
            train_loss = train_loss + loss.item()
        train_loss /= nbatch_train

        if i2 < ntrain-1 and ntrain < 100:
            x_ = x[i2:,:]
            y_ = y[i2:,:]
            x_ = x_.to(device)
            y_ = y_.to(device)
            opt.zero_grad()
            y_pred_ = model(x_)
            loss = criterion(y_, y_pred_)
            loss.backward()
            opt.step()
            train_loss = (train_loss*train_batch_size*nbatch_train + loss.item()*(ntrain-i2))/ntrain

        history['train_loss'].append(train_loss)


        val_loss = 0.0
        if nbatch_val>0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.
                for i in range(nbatch_val):
                    i1 = i*val_batch_size
                    i2 = i1+val_batch_size
                    x_ = val_x[i1:i2,:]
                    y_ = val_y[i1:i2,:]
                    x_=x_.to(device)
                    y_=y_.to(device)
                    y_pred_ = model(x_)
                    loss = criterion(y_, y_pred_)
                    val_loss += loss.item()
                val_loss /= nbatch_val

                if i2 < nval-1:
                    x_ = val_x[i2:,:]
                    y_ = val_y[i2:,:]
                    x_ = x_.to(device)
                    y_ = y_.to(device)
                    opt.zero_grad()
                    y_pred_ = model(x_)
                    loss = criterion(y_, y_pred_)
                    val_loss = (val_loss*val_batch_size*nbatch_val + loss.item()*(nval-i2))/nval
            history['val_loss'].append(val_loss)
            model.train()

            if val_loss < best:
                best = val_loss
                model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(model_state_dict,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        else:
            if train_loss < best:
                best = train_loss
                model_state_dict = copy(model.state_dict())
                opt_state_dict = copy(opt.state_dict())
                if epoch > save_epoch + 5:
                    save_epoch = epoch
                    torch.save(model_state_dict,fname_model)
                    torch.save(opt_state_dict, fname_opt)
                    pickle.dump(history,open(fname_history,'wb'))
        
        
        if verbose:
            nskip = int(epochs/2000)
            if epoch%nskip==0:
                elapsed_t = datetime.timedelta(seconds=time.monotonic() - t0)
                if dropout_stabilization:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | Val Loss: {val_loss:.2E} | lr: {lr_:.2E} |  dropout: {model.dropout_p:.2E} | {elapsed_t}')
                else:
                    print(f' Epoch {epoch+0:04}: | Train Loss: {train_loss:.2E} | Val Loss: {val_loss:.2E} | lr: {lr_:.2E} | {elapsed_t}')

    dt = time.monotonic()-t0
    if 0 < training_timeout < np.inf:
        new_epochs = min(epochs, training_timeout/(dt/(epoch-epoch_start+1)))
        if new_epochs != epochs:
            epochs = new_epochs
            if lr_scheduler:
                last_epoch = epoch*train_batch_size
                scheduler = torch.optim.lr_scheduler.OneCycleLR(max_lr=lr, div_factor=5, pct_start=0.1, final_div_factor=50,
                                                            epochs=epochs, steps_per_epoch=nbatch_train, last_epoch=last_epoch)
                
    if load_best:
        model.load_state_dict(model_state_dict)
            
    return history,model_state_dict,opt_state_dict



def format_floats(input_data, num_digits=4):
    """
    Format a single float or an array of floats by rounding to a specified number of significant digits.

    Parameters:
    - input_data (float, int, list, or numpy.ndarray): The input data to be formatted.
    - num_digits (int, optional): The number of significant digits. Defaults to 4.

    Returns:
    - float or numpy.ndarray: The formatted value(s).
    """

    if not isinstance(num_digits, int) or num_digits <= 0:
        raise ValueError("Number of significant digits must be a positive integer")

    if isinstance(input_data, (float, int)):
        # Round to the specified number of significant digits
        formatted_val = round(input_data, num_digits - int(np.floor(np.log10(abs(input_data)))) - 1)

        return float(formatted_val)
    elif isinstance(input_data, (list, np.ndarray)):
        # Round to the specified number of significant digits for each element
        formatted_arr = [round(val, num_digits - int(np.floor(np.log10(abs(val)))) - 1) for val in input_data]

        # Convert back to float for each element
        formatted_arr = np.array(formatted_arr, dtype=float)

        return formatted_arr
    else:
        raise ValueError("Input must be a float, int, list, or numpy array")
