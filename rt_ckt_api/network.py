import numpy as np
import skrf as rf
import numba as nb
from typing import Optional, Callable, Any, Sequence
from rt_math_api.utility import ExpressionValue, UNIT_TO_VALUE
import re

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


@nb.njit([nb.float64[:, :](nb.complex128[:, :, :]),
          nb.float64[:, :](nb.complex64[:, :, :]),])
def calculate_passivity_matrix(s_parameter: np.ndarray) -> np.ndarray:
    r'''
    Description:
    ------------
    Objective:
        Calculate passitivity matrix whose element is the sum of the power of the reflection coefficient of each port.
        
        For fixed frequency, $P_i = \sum_k |s_ik|^2$, where i = 0, 1, ..., n_port-1. 
    
    Benchmark:
        1. Platform
            CPU Model: Intel(R) Xeon(R) W-3245 CPU @ 3.20GHz
            CPU Arch: X86_64
            Memory: 383 GB
        2. Comparison
            | Target Size     | `calculate_passivity_matrix` | `rf.passivity`    |  Speedup         |
            | (440, 4, 4)     |    91.9 µs ± 80.3 ns         | 2.05 ms ± 4.67 µs |  22.3x           |
            | (360, 103, 103) |  22.3 ms ± 3.13 ms           | 340 ms ± 20.4 ms  |  15.3x           |            
            
    Parameters
    ----------
    s_parameter : np.ndarray
        The s-parameter matrix with the shape of (n_freq, n_port, n_port).
        
    Returns
    -------
    passivity_array : np.ndarray
        The passivity array with the shape of (n_freq, n_port).
        
    '''
    assert s_parameter.shape[1] == s_parameter.shape[2], 'The shape of s_parameter must be (n_freq, n_port, n_port).'
    
    passivity_array = np.zeros(s_parameter.shape[:2], dtype=np.float64)
    for i in range(s_parameter.shape[0]):
        gram_matrix = s_parameter[i] * s_parameter[i].conj()  #* G_ij = |S_ij|^2
        for j in range(s_parameter.shape[1]):
            passivity_array[i, j] = sum(gram_matrix[j, :].real)  #* sum of the row

    return passivity_array


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
        # 'tdr': [''],
        }
    SUPPORTED_NETWORK_DATA_PROPERTIES: list[str] = [f'{ntwk_type}_{data_type}' for ntwk_type, data_type_list in SUPPORTED_NETWORK_DATA.items() for data_type in data_type_list]
    
    
    N_FITTED = 2000 #* default number of points for vector fitting interpolation
    network: rf.Network
    fitted_network: rf.Network = None
    vector_fitting: rf.VectorFitting = None
    
    _tdr: np.ndarray = None
    _time: np.ndarray = None
    _passivity_matrix: np.ndarray = None
    _causality_matrix: np.ndarray = None
    
    def __init__(self, touchstone_filepath: str = '', s_paraeter: Optional[np.ndarray] = None, freq: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None, ):
        
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
        
        #* Load skrf.Network property
        self._dynamic_load_network_parameter(self.network)
    
    def _dynamic_load_network_parameter(self, network: rf.Network):
        '''Load the supported properties of skrf.Network to the NetworkData object.'''
        for attr in self.SUPPORTED_NETWORK_DATA_PROPERTIES + ['f']:
            skrf_ntwk_prop = getattr(network, attr, None)
            if skrf_ntwk_prop is None:
                raise ValueError(f'Property {attr} is not supported by skrf.Network.')
            setattr(self, attr, skrf_ntwk_prop)
        
        # for 
    
    def _load_touchstone(self, touchstone_filepath: str):
        
        
        
        comment_lines: list[str] = []
        option_line: str = None
        data: list[str] = []
        
        #$ Read the touchstone file
        with open(touchstone_filepath, 'r') as file_io:

            for line in file_io.readlines():
                line = line.strip()
                
                #* Deal with comment lines 
                if line.startswith('!'):
                    comment_lines += [line]
                    
                #* Deal with option line
                elif line.startswith('#'):
                    if option_line is not None:
                        raise ValueError('Multiple option lines are found in the touchstone file.')
                    
                    option_line = line
                    
                    _, freq_unit, ntwk_type, data_format, __, z_ref = re.findall(r'\S+', line)
                    if __ != 'R':
                        raise ValueError(f'Wierd option line: {line}')
                    
                    #* Check if the network type is S-parameter or not
                    if ntwk_type != 'S':
                        raise ValueError(f'Currently, only S-parameter is supported. Found {ntwk_type}-parameter.')
                    
                    #* Check if the data type is valid or not
                    if data_format.lower() not in ['ma', 'db', 'ri']:
                        raise ValueError(f'Invalid data type: {data_format}.')
                
                #* Deal with data
                else:
                    data += re.findall(r'\S+', line)
                    

        #$ Process parsed data
        #: 1. Parse the comment lines
        ansys_port_pattern = re.compile(r'Port\s+(\d+):\s+Z0\s+=\s+(\d+\.\d+)\s+Ohm')
        cadence_port_pattern = re.compile(r'POhm')
        
        
    def check_passivity(self) -> list[tuple[int, str, float]]:
        # if self._passivity_matrix is not None:
        passivity_matrix = calculate_passivity_matrix(self.s)

        passivity_reports: list = []
        for i in range(self.n_freq):
            for j in range(self.n_port):
                if passivity_matrix[i, j] > 1:
                    passivity_reports += [(i, str(ExpressionValue(self.f[i], unit='GHz')), passivity_matrix[i, j])]
        
        self._passivity_matrix = passivity_matrix
    
    def check_causality(self):
        pass
    
    def self_interpolate(self, n_fitted: Optional[int] = None):
        '''Interpolate the s parameters in the same range of loaded frequency but with linearly spaced frequency points.'''
        
        #* Prepare the new frequency and z0
        n_fitted = n_fitted if n_fitted is not None else self.N_FITTED
        new_z0 = np.tile(self.network.z0[0], (n_fitted, 1))
        n_port = self.network.nports
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
        _, step_func = generate_step_function(time[0], time[-1], len(time), rise_time=rise_time, delay=delay)
        print(impulse.shape, impulse[:, 0, 0].shape, step_func.shape)
        
        tdr = np.ones((time.shape[0], self.n_port), dtype=np.float64)
        for i in range(self.n_port):
            reflection_profile = np.convolve(impulse[:, i, i], step_func)
            
            n_convolve = len(reflection_profile)
            reflection_profile = reflection_profile[n_convolve//2-time.shape[0]//2: n_convolve//2+time.shape[0]//2].real #* 把合理的時間範圍取出來
            
            z0 = self.network.z0[0, i]
            tdr[:,i] = z0*(1+reflection_profile)/(1-reflection_profile)
        
        self._time = time
        self._tdr = tdr
        
        return time, tdr
    
    # def __getattribute__(self, name: str) -> Any:
    #     if name in self.SUPPORTED_NETWORK_DATA_PROPERTIES and name != 'tdr':
    #         return getattr(self.network, name)
    #     else:
    #         return getattr(self, name)
        
    def __getattr__(self, name: str) -> Any:
        
        return getattr(self.network, name, None) 
    
    def _load_s_parameters(self):
        pass



if '__main__' == __name__:
    
    import matplotlib.pyplot as plt
    import time
    t1 = time.time()
    path  = r'D:/Users/szuhsien.feng/Desktop/TEMP/RL7025_BGAc_AFE_Q3DSNP_20240708.s103p'
    ntwk_data = NetworkData(path)
    t2 = time.time()
    print(f'Loading takes {t2-t1}sec')
    
    time, tdr = ntwk_data.calculate_tdr()
    t3 = time.time
    print(f'TDR takes {t3-t2}sec')
    # for i in range(tdr.shape[1]):
    #     print(i)
    #     plt.plot(time, tdr[:, i])
    # plt.xlim(0, 10e-9)
    # print(f'Loading take {t2-t1}sec')
    #! Test the NetworkData class
    #* Load
