import numpy as np
import skrf as rf
import numba as nb
from typing import Optional, Callable, Any, Sequence

#%% Functions
@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(nb.float64, nb.float64, nb.int64, nb.float64, nb.float64))
def generate_step_function(start_time: int|float, end_time: int|float, num_points: int, rise_time: float = 50e-12, delay: float = 1e-9):
    """
    Generate a step function with specified rise time and delay.

    Parameters
    ----------
    start_time : float
        Start time of the step function.
    end_time : float
        End time of the step function.  
    num_points : int
        Number of points in the step function.
    rise_time : float, optional
        Rise time of the step function. The default is 50e-12.
    delay : float, optional
        Delay of the step function. The default is 1e-9.
        
    Returns
    -------
    t : np.ndarray
        Time vector.
    y : np.ndarray
        Step function.  
    """
    t = np.linspace(start_time, end_time, num_points)
    y = np.empty_like(t)
    
    for i in range(num_points):
        y[i] = min(max((t[i] - delay) / rise_time, 0), 1)
    
    return t, y


@nb.njit
def calculate_passivity_matrix(freq, s_parameter):
    pass

@nb.njit
def calculate_causality_matrix(freq, s_parameter, criteria):
    pass

#%% Objects


class NetworkData():
    
    #* Default parameters
    SUPPORTED_NETWORK_DATA: dict[str, list[str]] = {
        's': ['mag', 'db', 'deg', 're', 'im'],
        'y': ['mag', 'db', 'deg', 're', 'im'],
        'z': ['mag', 'db', 'deg', 're', 'im'],
        'tdr': [''],
        }
    SUPPORTED_NETWORK_DATA_PROPERTIES: list[str] = [f'{ntwk_type}_{data_type}' for ntwk_type, data_type_list in SUPPORTED_NETWORK_DATA.items() for data_type in data_type_list]
    
    
    N_FITTED = 2000 #* default number of points for vector fitting interpolation
    network: rf.Network
    fitted_network: rf.Networ
    vector_fitting: rf.VectorFitting
    
    _tdr: Sequence[float] 
    _time: Sequence[float]
    
    def __init__(self, touchstone_filepath: str = '', s_paraeter: Optional[np.ndarray] = None, freq: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None,  n_fitted: Optional[int] = None):
        
        self.n_fitted = n_fitted if n_fitted is not None else self.n_fitted
        
        if touchstone_filepath:
            self.network = rf.Network(touchstone_filepath)
        elif (s_paraeter is not None and 
             freq is not None and 
             z0 is not None):
            self.network = rf.Network(s=s_paraeter, f=freq, z0=z0)
        else:
            raise ValueError('Either touchstone_filepath or s_paraeter, freq, z0 must be provided.')
        
        
        #* Basic information
        self.n_freq = self.network.f.shape[0]
        self.n_port = self.network.nports
        self.filepath = touchstone_filepath
    
    def check_passivity(self):
        pass
    
    def check_causality(self):
        pass
    
    def self_interpolate(self, n_fitted: Optional[int] = None):
        '''Interpolate the s parameters in the same range of loaded frequency but with linearly spaced frequency points.'''
        
        #* Prepare the new frequency and z0
        n_fitted = n_fitted if n_fitted is not None else self.N_FITTED
        new_z0 = np.tile(self.network.z0[0], (n_fitted, 1))
        n_port = self.raw_network.nports
        freq = np.linspace(self.network.f.min(), self.network.f.max(), n_fitted)
        
        
        #* Interpolate the s parameters by vector fitting
        self.vector_fitting = rf.VectorFitting(self.network)            
        self.vector_fitting.auto_fit()
        s_param = np.ones((n_fitted, n_port, n_port), dtype=np.complex128)
        for i in range(n_port):
            for j in range(n_port):
                s_param[:,i,j] = self.vector_fitting.get_model_response(i, j, freq)
        
        self.fitted_network = rf.Network(s=s_param, f=freq, z0=new_z0)        
        
    
    
    
    
    def calculate_tdr(self, window:str = 'hamming', pad: int = 1000, rise_time: float = 50e-12, delay: float = 1e-9):
        #! 可以考慮用 non-uniform FFT 來計算 Impulse Response，然後再計算TDR。
        #! 如果準度可以接受，則可棄用 VectorFitting 來做 Interpolation。
        
        #* Check if the frequency sampling is uniform or not. If not, doing the vector fitting interpolation
        self.self_interpolate()
        
        #* Generate the step function
        time, impulse = self.fitted_network.impulse_response(window=window, pad=pad)
        step_func = generate_step_function(time[0], time[-1], len(time), rise_time=rise_time, delay=delay)
        
        
        tdr = np.ones((time.shape[0], self.n_port), dtype=np.float64)
        for i in range(self.n_port):
            reflection_profile = np.convolve(impulse[:, i, i], step_func)
            n_convolve = len(reflection_profile)
            reflection_profile = reflection_profile[n_convolve//2-time.shape[0]//2: n_convolve//2+time.shape[0]//2] #* 把合理的時間範圍取出來
            
            z0 = self.network.z0[0, i]
            tdr[:,i] = z0*(1+reflection_profile)/(1-reflection_profile)
        
        self._time = time
        self._tdr = tdr
        
        return time, tdr
    
    def __getattribute__(self, name: str) -> Any:
        if name in self.SUPPORTED_NETWORK_DATA_PROPERTIES and name != 'tdr':
            return getattr(self.network, name)
        else:
            return getattr(self, name)
        
    
    def _load_s_parameters(self):
        pass




