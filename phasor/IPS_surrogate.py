import IPS_surrogate_util as util
import torch
import os
import json
import numpy as np
from cav_info import cav_info as _info

_beam_frequency = 80.5e6
_wavelength = 299792458 / _beam_frequency

_Q=10
_A=100
_max_QoverA = 1/2


class _surrogate_model_loader:
    def __init__(self,path,cav_type,verbose=False):
        """
        Loads a surrogate model and corresponding scalar for a specific cavity type

        Args:
            path (str): Path to the directory containing the model files.
            cav_type (str): Cavity type (e.g., 'QWR029').
            verbose (bool, optional): Whether to print information input/output normlization
        """    
        scalar = util.scalar(fname = os.path.join(path,'scalar_info.pkl'))
        model_info = json.load(open( os.path.join(path,'model_info.json'),'r'))
        model, _ = util.construct_model(**model_info)
        model.to('cpu').to(torch.float32);
        model = torch.jit.script(model)
        model.load_state_dict(torch.load(
                                    os.path.join(path,'model.pt'),
                                    map_location=torch.device('cpu')))
        model.eval();
        self.scalar = scalar
        self.model = model
        self.cav_type = cav_type
        self.cav_length = _info[cav_type]['cav_length']
        if verbose:
            print(f"{cav_type} scalar info:",self.scalar.info)
        
    def __call__(self,phase,Wu,cav_amp,qA):
        """
        Make a prediction using the surrogate model.

        Parameters:
        - phase
        - Wu: W/u 
        - cav_amp: Cavity amplitude
        - qA: q/A

        Returns:
        - dTau: Prediction for dTau
        - dWu: Prediction for dWu
        """
        x = torch.stack([phase, Wu, cav_amp, qA], dim=1)
        if np.any(x[:,1:] < self.scalar.info['xmin'][:,1:]) or \
           np.any(x[:,1:] > self.scalar.info['xmin'][:,1:] + self.scalar.info['xdiff'][:,1:]):
            print(f"WARN! {self.cav_type} model input of q/A: {f'{qA}'} or cav_amp: {f'{cav_amp:.2e}'} is out-of-training distribution. Model prediction may not accurate")
            print(f"q/A min,max: {self.scalar.info['xmin'][0,3], self.scalar.info['xmin'][0,3]}")
            print(f"cav_amp min,max: {self.scalar.info['xmin'][0,2], self.scalar.info['xmin'][0,2]}")

        xn = self.scalar.normalize_x(x)
        yn = self.model(xn)
        y = self.scalar.unnormalize_y(yn)
        dTau = y[:,0]
        dWu  = y[:,1]
        return dTau, dWu


# def expand_inputs(phase,Wu,cav_amp,qA,dtype=torch.float32):
#     phase = torch.atleast_1d(phase).to(dtype)
#     Wu = torch.atleast_1d(Wu).to(dtype)
#     cav_amp = torch.atleast_1d(cav_amp).to(dtype)
#     qA = torch.atleast_1d(qA).to(dtype)

#     batch_size = max(phase.shape[0],Wu.shape[0],cav_amp.shape[0],qA.shape[0])
#     assert len(phase)  == batch_size or len(phase)  ==1
#     assert len(Wu)     == batch_size or len(Wu)     ==1
#     assert len(cav_amp)== batch_size or len(cav_amp)==1
#     assert len(qA)     == batch_size or len(qA)     ==1

#     # Broadcast Wu, cav_amp, qA to match phase length if they are scalars
#     if phase.numel() == 1:
#         phase = phase.expand(batch_size)
#     if Wu.numel() == 1:
#         Wu = Wu.expand(batch_size)
#     if cav_amp.numel() == 1:
#         cav_amp = cav_amp.expand(batch_size)
#     if qA.numel() == 1:
#         qA = qA.expand(batch_size)

#     return phase,Wu,cav_amp,qA


def expand_inputs(phase, Wu, cav_amp, qA, dtype=torch.float32):
    phase = torch.as_tensor(phase, dtype=dtype)
    Wu = torch.as_tensor(Wu, dtype=dtype)
    cav_amp = torch.as_tensor(cav_amp, dtype=dtype)
    qA = torch.as_tensor(qA, dtype=dtype)

    # Handle 2D inputs (n_scan, batch_size)
    if phase.ndim == 2:
        n_scan, batch_size = phase.shape
        assert Wu.shape == (n_scan, batch_size) or Wu.shape == (batch_size,) or Wu.shape == (), \
            f"Expected Wu shape {(n_scan, batch_size)}, {(batch_size,)}, or (), got {Wu.shape}"
        assert cav_amp.shape == (n_scan, batch_size) or cav_amp.shape == (batch_size,) or cav_amp.shape == (), \
            f"Expected cav_amp shape {(n_scan, batch_size)}, {(batch_size,)}, or (), got {cav_amp.shape}"
        assert qA.shape == (n_scan, batch_size) or qA.shape == (batch_size,) or qA.shape == (), \
            f"Expected qA shape {(n_scan, batch_size)}, {(batch_size,)}, or (), got {qA.shape}"

        # Broadcast scalars or 1D tensors to (n_scan, batch_size)
        if Wu.ndim < 2:
            Wu = Wu.expand(n_scan, batch_size) if Wu.ndim == 1 else Wu.expand(n_scan, batch_size)
        if cav_amp.ndim < 2:
            cav_amp = cav_amp.expand(n_scan, batch_size) if cav_amp.ndim == 1 else cav_amp.expand(n_scan, batch_size)
        if qA.ndim < 2:
            qA = qA.expand(n_scan, batch_size) if qA.ndim == 1 else qA.expand(n_scan, batch_size)
    else:
        # Original 1D input handling
        phase = torch.atleast_1d(phase)
        Wu = torch.atleast_1d(Wu)
        cav_amp = torch.atleast_1d(cav_amp)
        qA = torch.atleast_1d(qA)
        batch_size = max(phase.shape[0], Wu.shape[0], cav_amp.shape[0], qA.shape[0])
        assert phase.shape[0] == batch_size or phase.shape[0] == 1
        assert Wu.shape[0] == batch_size or Wu.shape[0] == 1
        assert cav_amp.shape[0] == batch_size or cav_amp.shape[0] == 1
        assert qA.shape[0] == batch_size or qA.shape[0] == 1
        if phase.shape[0] == 1:
            phase = phase.expand(batch_size)
        if Wu.shape[0] == 1:
            Wu = Wu.expand(batch_size)
        if cav_amp.shape[0] == 1:
            cav_amp = cav_amp.expand(batch_size)
        if qA.shape[0] == 1:
            qA = qA.expand(batch_size)
        n_scan = 1

    return phase, Wu, cav_amp, qA

        
_script_dirname = os.path.dirname(os.path.abspath(__file__))
class _combine_multiInputDomainWu_surrogate_models:
    def __init__(self,cav_type):  
        """
        Initialize the combined (split based on input Wu domain) surrogate models.

        Parameters:
        - cav_type: Type of cavity
        """
        self.cav_type = cav_type
        self.cav_length = _info[cav_type]['cav_length']
        self.cav_frequency = _info[cav_type]['cav_frequency']
        self.nLEVEL = _info[cav_type]['nLEVEL']
        self.W_u_range = _info[cav_type]['W_u_range']
        self.models = [
            _surrogate_model_loader(
                os.path.join(_script_dirname,cav_type,'WuLEVEL'+str(i)), 
                cav_type)  
            for i in range(self.nLEVEL)]
        

    def __call__(self, phase, Wu, cav_amp, qA):
        phase, Wu, cav_amp, qA = expand_inputs(phase, Wu, cav_amp, qA)
        if phase.ndim == 2:  # (n_scan, batch_size)
            n_scan, batch_size = phase.shape
            # Flatten to (n_scan * batch_size,) for model processing
            phase_flat = phase.reshape(-1)
            Wu_flat = Wu.reshape(-1)
            cav_amp_flat = cav_amp.reshape(-1)
            qA_flat = qA.reshape(-1)
            x_flat = torch.stack([phase_flat, Wu_flat, cav_amp_flat, qA_flat], dim=1)  # (n_scan * batch_size, 4)
            
            n = torch.clamp(
                self.nLEVEL * (Wu_flat - self.W_u_range[0]) / (self.W_u_range[1] - self.W_u_range[0]),
                min=0, max=self.nLEVEL - 1
            ).long()
            
            dTau = torch.zeros_like(phase_flat)
            dWu_out = torch.zeros_like(phase_flat)
            
            for i in range(self.nLEVEL):
                mask = (n == i)
                if mask.any():
                    model = self.models[i]
                    x_subset = x_flat[mask]
                    if isinstance(model.scalar.info['xmin'], torch.Tensor):
                        xmin = model.scalar.info['xmin'][:, 1:]
                    else:
                        xmin = torch.tensor(model.scalar.info['xmin'][:, 1:], dtype=torch.float32)
                    if isinstance(model.scalar.info['xdiff'], torch.Tensor):
                        xdiff = model.scalar.info['xdiff'][:, 1:]
                    else:
                        xdiff = torch.tensor(model.scalar.info['xdiff'][:, 1:], dtype=torch.float32)
                    mask_domain = torch.logical_or(
                        x_subset[:, 1:].lt(xmin).any(dim=1),
                        x_subset[:, 1:].gt(xmin + xdiff).any(dim=1)
                    )
                    if mask_domain.any():
                        print(f"WARN! {self.cav_type} model input (q/A or cav_amp) for WuLEVEL{i} is out-of-training distribution.")
                        print(f"Wu min, max: {model.scalar.info['xmin'][0,1]}, {model.scalar.info['xmin'][0,1] + model.scalar.info['xdiff'][0,1]}")
                        print(f"cav_amp min, max: {model.scalar.info['xmin'][0,2]}, {model.scalar.info['xmin'][0,2] + model.scalar.info['xdiff'][0,2]}")
                        print(f"q/A min, max: {model.scalar.info['xmin'][0,3]}, {model.scalar.info['xmin'][0,3] + model.scalar.info['xdiff'][0,3]}")
                        print(f'x_subset[mask_domain]', x_subset[mask_domain, 1:])
                    
                    xn = model.scalar.normalize_x(x_subset)
                    yn = model.model(xn)
                    y = model.scalar.unnormalize_y(yn)
                    dTau[mask] = y[:, 0]
                    dWu_out[mask] = y[:, 1]
            
            # Reshape back to (n_scan, batch_size)
            dTau = dTau.reshape(n_scan, batch_size)
            dWu_out = dWu_out.reshape(n_scan, batch_size)
        else:
            # Original 1D input handling
            n = torch.clamp(
                self.nLEVEL * (Wu - self.W_u_range[0]) / (self.W_u_range[1] - self.W_u_range[0]),
                min=0, max=self.nLEVEL - 1
            ).long()
            x = torch.stack([phase, Wu, cav_amp, qA], dim=1)
            dTau = torch.zeros_like(phase)
            dWu_out = torch.zeros_like(phase)
            for i in range(self.nLEVEL):
                mask = (n == i)
                if mask.any():
                    model = self.models[i]
                    x_subset = x[mask]
                    # ... (same domain checks as above)
                    xn = model.scalar.normalize_x(x_subset)
                    yn = model.model(xn)
                    y = model.scalar.unnormalize_y(yn)
                    dTau[mask] = y[:, 0]
                    dWu_out[mask] = y[:, 1]
        
        return dTau, dWu_out
 
    # def __call__(self,phase,Wu,cav_amp,qA):
    #     """
    #     Make a prediction using the combined surrogate models.

    #     Parameters:
    #     - phase
    #     - Wu: W/u 
    #     - cav_amp: Cavity amplitude
    #     - qA: q/A

    #     Returns:
    #     - dTau: Prediction for dTau
    #     - dWu: Prediction for dWu
    #     """
    #     # phase = torch.atleast_1d(phase).to(torch.float32)
    #     # Wu = torch.atleast_1d(Wu).to(torch.float32)
    #     # cav_amp = torch.atleast_1d(cav_amp).to(torch.float32)
    #     # qA = torch.atleast_1d(qA).to(torch.float32)

    #     # # Broadcast Wu, cav_amp, qA to match phase length if they are scalars
    #     # if Wu.numel() == 1:
    #     #     Wu = Wu.expand_as(phase)
    #     # if cav_amp.numel() == 1:
    #     #     cav_amp = cav_amp.expand_as(phase)
    #     # if qA.numel() == 1:
    #     #     qA = qA.expand_as(phase)
        
    #     # Compute model indices based on Wu (non-differentiable)

    




    #     phase,Wu,cav_amp,qA = expand_inputs(phase,Wu,cav_amp,qA)
    #     n = torch.clamp(
    #         self.nLEVEL * (Wu - self.W_u_range[0]) / (self.W_u_range[1] - self.W_u_range[0]),
    #         min=0, max=self.nLEVEL - 1
    #     ).long()
    #     x = torch.stack([phase, Wu, cav_amp, qA], dim=1)

    #     # Initialize output tensors
    #     dTau = torch.zeros_like(phase)
    #     dWu_out = torch.zeros_like(phase)

    #     # Batch process all models
    #     xn = torch.cat([model.scalar.normalize_x(x) for model in self.models], dim=0)
    #     yn = torch.cat([model.model(xn[i*x.shape[0]:(i+1)*x.shape[0]]) for i, model in enumerate(self.models)], dim=0)
    #     y = torch.cat([model.scalar.unnormalize_y(yn[i*x.shape[0]:(i+1)*x.shape[0]]) for i, model in enumerate(self.models)], dim=0)
        
    #     for i in range(self.nLEVEL):
    #         mask = (n == i)
    #         if mask.any():
    #             idx = i * x.shape[0]
    #             dTau[mask] = y[idx:idx+x.shape[0], 0][mask]
    #             dWu_out[mask] = y[idx:idx+x.shape[0], 1][mask]
    #     return dTau, dWu_out


        # # Process each model for corresponding inputs
        # for i in range(self.nLEVEL):
        #     mask = (n == i)
        #     if mask.any():
        #         model = self.models[i]
        #         x_subset = x[mask]
        #         # Check if inputs are within training distribution 
        #         if isinstance(model.scalar.info['xmin'], torch.Tensor):
        #             xmin = model.scalar.info['xmin'][:,1:]
        #         else:
        #             xmin = torch.tensor(model.scalar.info['xmin'][:,1:], dtype=torch.float32)
        #         if isinstance(model.scalar.info['xdiff'], torch.Tensor):
        #             xdiff = model.scalar.info['xdiff'][:,1:]
        #         else:
        #             xdiff = torch.tensor(model.scalar.info['xdiff'][:,1:], dtype=torch.float32)
        #         mask_domain = torch.logical_or(x_subset[:, 1:].lt(xmin).any(dim=1), x_subset[:, 1:].gt(xmin + xdiff).any(dim=1))
        #         if mask_domain.any():
        #             print(f"WARN! {self.cav_type} model input (q/A or cav_amp) for WuLEVEL{i} is out-of-training distribution. Model prediction may not be accurate")
        #             print(f"Wu min, max:  {model.scalar.info['xmin'][0,1]}, {model.scalar.info['xmin'][0,1] + model.scalar.info['xdiff'][0,1]}")
        #             print(f"cav_amp min, max: {model.scalar.info['xmin'][0,2]}, {model.scalar.info['xmin'][0,2] + model.scalar.info['xdiff'][0,2]}")
        #             print(f"q/A min, max: {model.scalar.info['xmin'][0,3]}, {model.scalar.info['xmin'][0,3] + model.scalar.info['xdiff'][0,3]}")
        #             print(f'x_subset[mask_domain',x_subset[mask_domain,1:])

        #         # Normalize, predict, and unnormalize
        #         xn = model.scalar.normalize_x(x_subset)
        #         yn = model.model(xn)
        #         y = model.scalar.unnormalize_y(yn)

        #         # Store results
        #         dTau[mask] = y[:, 0]
        #         dWu_out[mask] = y[:, 1]

        # return dTau, dWu_out

        
QWR029 = _combine_multiInputDomainWu_surrogate_models('QWR029')
QWR053 = _combine_multiInputDomainWu_surrogate_models('QWR053')
QWR041 = _combine_multiInputDomainWu_surrogate_models('QWR041')
QWR085 = _combine_multiInputDomainWu_surrogate_models('QWR085')
