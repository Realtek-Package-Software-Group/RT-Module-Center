from datetime import datetime
# t1 = datetime.now()


import numpy as np
import skrf as rf
import numba as nb
import numpy as np
from typing import Optional, Callable, Any, Sequence
from pathlib import Path
from datetime import datetime
import time
import subprocess


from rt_math_api.utility import ExpressionValue, UNIT_TO_VALUE
from rt_nx_api.eda import CMDThread
from rt_nx_api.eda import LicenseClient

import re
import os
import platform
import threading
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")


try:
    import ctypes
    import msvcrt
except:
    pass

# t2 = datetime.now()

#%% Math Functions

nb.set_num_threads(os.cpu_count()//2) # type:ignore #: Set half of the CPU count as the numba threads



# 定義 njit 裝飾的函數
@nb.njit([nb.complex128(nb.float64, nb.float64), nb.complex128[:](nb.float64[:], nb.float64[:]),])
def ma2cmplx(mag: float|np.ndarray, ang: float|np.ndarray) -> complex|np.ndarray:
    real = mag * np.cos(np.deg2rad(ang))
    imag = mag * np.sin(np.deg2rad(ang))
    return real+1j*imag

@nb.njit([nb.complex128(nb.float64, nb.float64), nb.complex128[:](nb.float64[:], nb.float64[:]),])
def db2cmplx(db: float|np.ndarray, ang: float|np.ndarray) -> complex|np.ndarray:
    mag = 10**(db/20)
    real = mag * np.cos(np.deg2rad(ang))
    imag = mag * np.sin(np.deg2rad(ang))
    return real+1j*imag

@nb.njit([nb.complex128(nb.float64, nb.float64), nb.complex128[:](nb.float64[:], nb.float64[:]),])
def ri2cmplx(real: float|np.ndarray, imag: float|np.ndarray) -> complex|np.ndarray:
    return real + 1j*imag

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


def s2z() -> np.ndarray: ...

def s2y() -> np.ndarray: ...


@nb.njit([nb.float64[:, :](nb.complex128[:, :, :]), nb.float64[:, :](nb.complex64[:, :, :]),],  parallel=True)
def calculate_passivity_matrix(s_parameter: np.ndarray) -> np.ndarray:

    assert s_parameter.shape[1] == s_parameter.shape[2], 'The shape of s_parameter must be (n_freq, n_port, n_port).'
    
    passivity_array = np.zeros(s_parameter.shape[:2], dtype=np.float64)
    for i in nb.prange(s_parameter.shape[0]):
        s = s_parameter[i]
        gram_matrix = s @ s.conj().T
        passivity_array[i] = np.linalg.eigvals(gram_matrix).real

    return passivity_array



#%% Other Functions


def get_ansys_newest_version() -> tuple[str, str]:
    # Determine the system's platform
    sys_platform = platform.system()
    if sys_platform not in ['Linux', 'Windows']:
        raise Exception(f'Unsupported OS: {sys_platform}')

    # Define the base directory path for ANSYS installations
    if sys_platform == 'Linux':
        ANSYS_BASE_DIRPATH = r'/home/STools/Ansys/AnsysEM'.replace('\\', '/')
        version_regex = r'(\d{4})R(\d)'
    elif sys_platform == 'Windows':
        from termcolor import colored
        import socket
        ANSYS_BASE_DIRPATH = r'C:\Program Files\AnsysEM'
        version_regex = r'v(\d{3})'
        hostname = socket.gethostname()
        if hostname not in ['R072720101', '']:
            raise Exception(f'Unknown Hostname: {hostname}')
    
    if not Path(ANSYS_BASE_DIRPATH).exists():
        raise FileNotFoundError(f'ANSYS base directory cannot be found: {ANSYS_BASE_DIRPATH}')
    
    # List directories and extract versions using regex
    dirs = (name for name in os.listdir(ANSYS_BASE_DIRPATH) if os.path.isdir(os.path.join(ANSYS_BASE_DIRPATH, name)))
    version_dirs = [(name, ''.join(re.search(version_regex, name).groups()))
                    for name in dirs if re.search(version_regex, name)]

    # Sort and get the latest version directory
    if not version_dirs:
        raise Exception("No valid ANSYS installations found")
    
    latest_version_dirname = sorted(version_dirs, key=lambda x: x[1])[-1][0]

    # Construct the path to the latest version
    if sys_platform == 'Linux':
        year, release = re.findall(r'\d+', latest_version_dirname)
        latest_version_path = os.path.join(ANSYS_BASE_DIRPATH, latest_version_dirname, f'v{year[2:]}{release}', 'Linux64')
        latest_version_root = f'ANSYSEM_ROOT{year[2:]}{release}'
    else:
        latest_version_path = os.path.join(ANSYS_BASE_DIRPATH, latest_version_dirname, 'Win64')
        latest_version_root = f'ANSYSEM_ROOT{latest_version_dirname[1:]}'
    
    # print(version_dirs)  # Optional: print all detected versions
    return latest_version_path, latest_version_root

def check_causality_by_genequiv(touchstone_filepath: str, causality_tolerance: float = 0.01, cpu_count: int = -1):
    
    latest_version_path, latest_version_root = get_ansys_newest_version()
    serial_number = datetime.now().strftime('%Y_%m%d_%H%M%S')
    
    if platform.system() == 'Windows':
        genequiv_exepath: Path = Path(latest_version_path) / 'genequiv.exe' 
    elif platform.system() == 'Linux':
        genequiv_exepath: Path = Path(latest_version_path) / 'genequiv'
    else:
        raise Exception(f'Unsupported OS: {platform.system()}')
    
    filepath: Path = Path(touchstone_filepath)
    filename = filepath.name
    if not filepath.exists():
        raise FileNotFoundError(f'Touchstone cannot be found: {filepath}')
    
    #* Copy the touchstone file to the temporary directory
    # copied_filepath: Path = Path(filepath.parent / f'.CheckTouchstone_{serial_number}' / filepath.name)
    copied_filepath: Path = Path(os.path.expanduser(f'~/.CheckTouchstone_{serial_number}')) / filepath.name
    if not copied_filepath.parent.exists():
        copied_filepath.parent.mkdir(parents=True)
    shutil.copy(str(filepath), str(copied_filepath))
    
    #* Prepare the command flags
    cmd_list: list[str] = [f'"{genequiv_exepath}"']
    cpu_count = os.cpu_count()//2 if cpu_count < 0 else cpu_count # type:ignore
    cmd_list += ['-checkcausality', #: Check causality
                 '-causality_plots', #: Generate reconstructed touchstone and error/truncation bound touchstone
                 f'-causality_tol {causality_tolerance}', #: Causality tolerance
                 f'-mp {cpu_count}', #: Multi-processing number
                 # f'-prof {copied_filepath.parent/"log.txt"}', #: Profiling
                 f'-cccontinuation 0', 
                 f'-ccinterp 1', #: Interpolation method (猜測NDE預設)
                 f'-ccintegration 2', #: Integration method (猜測NDE預設)
                 f'-i "{copied_filepath}"' #: Input touchstone filepath
                 ] 
    
    #* Need to  change director
    rt_license_client = LicenseClient()
    
    if platform.system() == 'Windows':
        command = f'cd "{copied_filepath.parent}"&' + ' '.join(cmd_list)
    else:
        command = (f'setenv ANSYSLMD_LICENSE_FILE {":".join(rt_license_client.ANSYS_LICENSE_SERVER_LIST)};' 
                   + f'cd "{copied_filepath.parent}";' 
                   + ' '.join(cmd_list))
        command = f'csh -c \'{command}\''

    
    # print('\n')
    # print(command)
    # print('\n')
    thread = CMDThread(command, show_cmd_output=False)#: Command 預設是Bash
    thread.start()
    thread.join()
    while True:
        if thread.cmd_process.poll() is not None:
            break
        
    # time.sleep(1)
    
    #: Load the causality information (reconstructed s-parameter and discretization/truncation error bound )
    causality_infomation: dict[str, rf.Network] = {}
    for name in os.listdir(copied_filepath.parent): # type:ignore
        if 'DiscErrBnd' in name:
            causality_infomation['discretization_error'] = rf.Network(str(copied_filepath.parent / name))
        elif 'ReconsData' in name:
            causality_infomation['reconstructed_data'] = rf.Network(str(copied_filepath.parent / name))
        elif 'TruncErrBnd' in name:
            causality_infomation['truncation_error'] = rf.Network(str(copied_filepath.parent / name))
        
    if set(causality_infomation.keys()) != {'discretization_error', 'reconstructed_data', 'truncation_error'}:
        # print(' )
        raise ValueError('Causality check failed.' + 'Information to debug: \n' + command)
    
    
    #* Delete the copied touchstone file
    shutil.rmtree(copied_filepath.parent)
    
    return causality_infomation
    
    
# t3 = datetime.now()

#%% Objects
class NetworkData():
    
    #* Default parameters
    SUPPORTED_NETWORK_DATA: dict[str, list[str]] = {
        's': ['mag', 'db', 'deg', 're', 'im'],
        # 'y': ['mag', 'db', 'deg', 're', 'im'],
        # 'z': ['mag', 'db', 'deg', 're', 'im'],
        # 'tdr': [''],
        }
    SUPPORTED_NETWORK_DATA_PROPERTIES: list[str] = [f'{ntwk_type}_{data_type}' for ntwk_type, data_type_list in SUPPORTED_NETWORK_DATA.items() for data_type in data_type_list] + list(SUPPORTED_NETWORK_DATA)
    
    N_FITTED = 2000 #* default number of points for vector fitting interpolation
    network: rf.Network
    fitted_network: rf.Network = None
    vector_fitting: rf.VectorFitting = None
    
    #* Copied from skrf.Network
    f: np.ndarray = None
    s: np.ndarray = None
    port_names: Sequence[str] = None
    port_modes: Sequence[str] = None
    
    _tdr_window: str = None
    _tdr_rise_time: float = None
    _tdr_delay: float = None
    _tdr_time: np.ndarray = None
    _tdr: np.ndarray = None
    
    #* Passivity
    _is_passive: bool = None
    _passivity_matrix: np.ndarray = None
    _passivity_reports: list[dict] = None
    
    #* Causality
    _is_causal: bool = None
    _reconstructed_network: rf.Network = None
    _truncation_error_network: rf.Network = None
    _discretization_error_network: rf.Network = None
    _causality_matrix: np.ndarray = None
    _causality_reports: list[dict] = None
    
    #* Reciprocity
    _reciprocity_matrix: np.ndarray = None
    _reciprocity_reports: list[dict] = None
    _is_reciprocal: bool = None
    
    def __init__(self, touchstone_filepath: str = '', s_parameter: Optional[np.ndarray] = None, freq: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None, ):
        
        #* Load the raw network data
        if touchstone_filepath:
            ts = rf.Touchstone(touchstone_filepath)
            f: np.ndarray = ts.f
            s: np.ndarray = ts.s
            n_freq: int = s.shape[0]
            n_port: int = s.shape[1]
            reference_impedance:int|float = ts.resistance # type:ignore
            frequency_unit = next((unit for unit in UNIT_TO_VALUE if ts.frequency_unit.upper()==unit.upper()), 'GHz')
            
            #* Check if `reference_impedance` is wrongly parsed 
            expected_z0 = np.tile(reference_impedance, (n_freq, n_port))  # type:ignore
            if not (ts.z0 == expected_z0).all():
                print(f'Warning: touchstone "{touchstone_filepath}" seems parse wrong reference impedance. Corrected to {reference_impedance}.')
            
            self.reference_impedance = reference_impedance
            self.frequency_unit = frequency_unit
            self.network = rf.Network(s=s, f=f, z0=expected_z0)
            self.network.comments = ts.get_comments()
            self.network.comments_after_option_line = ts.comments_after_option_line
            self.network.variables = ts.get_comment_variables # type:ignore
            self.network.port_names = ts.port_names
            self.network.port_modes = ts.port_modes
            
            # print('self.network.z0', self.network.z0)
            
            try:
                self.network.name = os.path.basename(os.path.splitext(touchstone_filepath)[0])
            except:
                self.network.name = os.path.basename(os.path.splitext(ts.filename)[0])
            
        elif (s_parameter is not None and 
             freq is not None and 
             z0 is not None):
            self.network = rf.Network(s=s_parameter, f=freq, z0=z0)
        else:
            raise ValueError('Either touchstone_filepath or s_paraeter, freq, z0 must be provided.')
        
        #* Basic informati
        self.n_freq = self.network.f.shape[0]
        self.n_port = self.network.nports
        self.filepath = touchstone_filepath
        
        #* Load skrf.Network property
        self._dynamic_load_network_parameter(self.network)
    
    def _dynamic_load_network_parameter(self, network: rf.Network):
        '''Load the supported properties of skrf.Network to the NetworkData object.'''
        for attr in self.SUPPORTED_NETWORK_DATA_PROPERTIES + ['f', 'port_names', 'port_modes', ]:
            skrf_ntwk_prop = getattr(network, attr, None)
            if skrf_ntwk_prop is not None:
                setattr(self, attr, skrf_ntwk_prop)

    @property
    def is_reciprocal(self) -> bool:
        if self._is_reciprocal is None:
            self.check_reciprocity()
        return self._is_reciprocal
    
    @property
    def reciprocity_matrix(self) -> np.ndarray:
        '''Return the reciprocity matrix, where 2 -> reciprocal; 1 -> almost; 0 -> non-reciprocal.'''
        if self._reciprocity_matrix is None:
            self.check_reciprocity()
        return self._reciprocity_matrix
    
    @property
    def reciprocity_reports(self) -> list[dict]:
        if self._reciprocity_reports is None:
            self.check_reciprocity()
        return self._reciprocity_reports
    
    def check_reciprocity(self, abs_error: float = 1e-9, rel_error_min: float = 0, rel_error_max: float = 0.05) -> bool:
        '''Check if the network is reciprocal or not.'''
        if self._is_reciprocal is None:
            # self._is_reciprocal = self.network.is_reciprocal()
            reciprocity_matrix: np.ndarray = np.zeros_like(self.network.s, dtype=np.int64) #: 2 -> reciprocal; 1 -> almost; 0 -> non-reciprocal
            reciprocity_reports: list[dict] = []
            for k, s in enumerate(self.network.s):
                reciprocal: dict[tuple[int, int], tuple[float, float]] = {}
                nonreciprocal: dict[tuple[int, int], tuple[float, float]] = {}
                almost: dict[tuple[int, int], tuple[float, float]] = {}
                
                transposed_s = s.transpose()
                delta_s = s - transposed_s
                for i in range(self.n_port):
                    for j in range(i, self.n_port):
                        #: 1. Use absolute error to judge
                        #: 2. If failing, to use relative error to judge
                        rel_err = abs(delta_s[i, j]) / max(abs(s[i, j]), abs(s[j, i]))
                        abs_err = abs(delta_s[i, j])
                        if abs(delta_s[i, j]) <= abs_error or rel_err <= rel_error_min:
                            reciprocity_matrix[k, i, j] = 2
                            reciprocal[(i, j)] = (abs_err, rel_err)
                        elif rel_error_min < rel_err <= rel_error_max:
                            reciprocity_matrix[k, i, j] = 1
                            almost[(i, j)] = (abs_error, rel_err)
                        else:
                            reciprocity_matrix[k, i, j] = 0
                            nonreciprocal[(i, j)] = (abs_err, rel_err)
                        reciprocity_matrix[k, j, i] = reciprocity_matrix[k, i, j]
                reciprocity_reports += [{'index':k, 
                                         'freq': ExpressionValue(self.f[k], unit='Hz').string, 
                                         'reciprocal': reciprocal, 
                                         'nonreciprocal': nonreciprocal, 
                                         'almost': almost}]
            
            #* If zero in reciprocity_matrix, give False to _is_reciprocal
            self._reciprocity_matrix = reciprocity_matrix
            self._reciprocity_reports = reciprocity_reports
            self._is_reciprocal = (reciprocity_matrix != 0).all()

        return self._is_reciprocal
    
    def single_to_mixmode(self, ):
        '''Convert the single-ended network to mixed-mode network.'''
        if self.n_port % 2 == 1:
            raise ValueError('Currently, even port number is required for mixed-mode network.')
        

        n_channel = self.n_port // 2
        self.network.se2gmm(p=n_channel)
        
        if self.fitted_network:
            self.fitted_network.se2gmm(p=n_channel)
        
        #* Reset passivty/causality/reciprocity check
        self._is_reciprocal = None
        self._is_passive = None
        self._is_causal = None
        self._passivity_matrix = None
        self._passivity_reports = None
        self._causality_matrix = None
        self._causality_reports = None
        
        self._dynamic_load_network_parameter(self.network) #* Reload the network parameters

    @property
    def is_passive(self) -> bool:
        if self._is_passive is None:
            self.check_passivity()
        return self._is_passive
    
    @property
    def passivity_matrix(self):
        '''Return the passivity matrix whose element is the sum of the power of the reflection coefficient of each port.'''
        if self._passivity_matrix is None:
            self.check_passivity()
        return self._passivity_matrix

    @property
    def passivity_reports(self) -> list[dict]:
        '''
        Return the passivity reports.
        Item-Dictionary: {'index':frequecny index, 'freq': frequency string, 'nonpassive': [(port index, power summation), ...]}
        '''
        if self._passivity_reports is None:
            self.check_passivity()
        return self._passivity_reports
    
    def check_passivity(self) -> bool:
        passivity_matrix = calculate_passivity_matrix(self.network.s)

        passivity_reports: list[dict] = []
        for i in range(self.n_freq):
            nonpassive: dict[int, float] = {}
            passive: dict[int, float] = {}
            for j in range(self.n_port):
                if passivity_matrix[i, j] > 1:
                    nonpassive[j] = passivity_matrix[i, j]
                else:
                    passive[j] = passivity_matrix[i, j]
                    
            passivity_reports += [{'index':i, 
                                   'freq': ExpressionValue(self.f[i], unit='Hz').string, 
                                   'nonpassive': nonpassive, 
                                   'passive': passive}]
        
        self._passivity_matrix = passivity_matrix
        self._passivity_reports = passivity_reports
        self._is_passive = not (passivity_matrix > 1).any()

        return self._is_passive
    
    @property
    def is_causal(self) -> bool:
        if self._is_causal is None:
            self.check_causality()
        return self._is_causal
    
    @property
    def causality_matrix(self) -> np.ndarray:
        '''Return the causality matrix, where 2 -> causal; 1 -> inconclusive; 0 -> non-causal.'''
        if self._causality_matrix is None:
            self.check_causality()
        return self._causality_matrix
    
    @property
    def causality_reports(self) -> list[dict]:
        '''
        Return the causality reports.
        Item-dictionary: {'index': frequency index, 'freq': frequency string, 'inconclusive': [(port index, port index), ...], 'noncausal': [(port index, port index), ...]}
        '''
        if self._causality_reports is None:
            self.check_causality()
        return self._causality_reports
    
    def check_causality(self, causality_tolerance: float = 0.01, cpu_count: int = -1) -> bool:
        causality_infomation = check_causality_by_genequiv(self.filepath, causality_tolerance=causality_tolerance, cpu_count=cpu_count)
        self._reconstructed_network = causality_infomation['reconstructed_data']
        self._truncation_error_network = causality_infomation['truncation_error']
        self._discretization_error_network = causality_infomation['discretization_error']
        
        reconstructed_error: np.ndarray = abs(self.network.s - self._reconstructed_network.s)
        truncation_error: np.ndarray = abs(self._truncation_error_network.s[:,0,0])
        discretization_error: np.ndarray = abs(self._discretization_error_network.s)
        
        causality_matrix: np.ndarray = np.zeros_like(self.network.s, dtype=np.int64) #: 2 -> causal; 1 -> inconclusive; 0 -> non-causal
        causality_reports: list = []
        for i in range(self.n_freq):
            # _ith_report: dict = {'index':i, 'freq': str(ExpressionValue(self.f[i], unit='Hz').string), 'inconclusive': [],  'noncausal': []}
            inconclusive: dict[tuple[int, int], float] = {}
            noncausal: dict[tuple[int, int], float] = {}
            causal: dict[tuple[int, int], float] = {}
            for j in range(self.n_port):
                second_port_index_range: list = [0, self.n_port] if not self.is_reciprocal else [j, self.n_port]
                for k in range(*second_port_index_range):
                    error_bound = discretization_error[i,j,k] + truncation_error[i]
                    if error_bound > causality_tolerance:
                        causality_matrix[i,j,k] = 1  
                        inconclusive[(j, k)] = causality_tolerance - error_bound
                    elif abs(reconstructed_error[i,j,k]) > causality_tolerance:
                        causality_matrix[i,j,k] = 0
                        noncausal[(j, k)] = causality_tolerance - abs(reconstructed_error[i,j,k])
                    else:
                        causality_matrix[i,j,k] = 2
                        causal[(j, k)] = causality_tolerance - abs(reconstructed_error[i,j,k])
            causality_reports += [{'index':i, 
                                   'freq': str(ExpressionValue(self.f[i], unit='Hz').string), 
                                   'inconclusive': inconclusive, 
                                   'noncausal': noncausal, 
                                   'causal': causal}]
        

        self._causality_matrix = causality_matrix
        self._causality_reports = causality_reports
        self._is_causal = not (causality_matrix != 2).any()
        return self._is_causal
    
    def linearly_interpolate(self):
        # '''Interpolate the s parameters in the same range of loaded frequency but with linearly spaced frequency points.'''
        '''
        Interpolate the s parameters in the same range of loaded frequency but with linearly spaced frequency points.
        
        Assume last band of extracting frequency is linearly spaced.
        '''
        #* Prepare the new frequency and z0
        lin_delf = np.diff(self.network.f)[-1]
        freq = np.arange(0, self.network.f.max(), lin_delf)
        self.fitted_network = self.network.interpolate(freq, basis='s', coords='cart') # type: ignore
        
        # n_fitted = n_fitted if n_fitted is not None else self.N_FITTED
        # new_z0 = np.tile(self.network.z0[0], (n_fitted, 1))
        # n_port = self.network.nports
        # freq = np.linspace(self.network.f.min(), self.network.f.max(), n_fitted)
        
        
        # #* Interpolate the s parameters by vector fitting
        # self.vector_fitting = rf.VectorFitting(self.network)            
        # self.vector_fitting.auto_fit()
        # s_param = np.ones((n_fitted, n_port, n_port), dtype=np.complex128)
        # for i in range(n_port):
        #     for j in range(n_port):
        #         s_param[:,i,j] = self.vector_fitting.get_model_response(i, j, freq)
        
        # self.fitted_network = rf.Network(s=s_param, f=freq, z0=new_z0)        
        
    def calculate_tdr(self, window: str = '', pad: int = 1000, rise_time: float = 50e-12, delay: float = 1e-9):
        #! 可以考慮用 non-uniform FFT 來計算 Impulse Response，然後再計算TDR。
        #! 如果準度可以接受，則可棄用 VectorFitting 來做 Interpolation。
        
        
        #* Check if the frequency sampling is uniform or not. If not, doing the vector fitting interpolation
        self.linearly_interpolate()
        
        #* Generate the step function
        time, impulse = self.fitted_network.impulse_response(window=None if not window else window, pad=pad)
        _, step_func = generate_step_function(time[0], time[-1], len(time), rise_time=rise_time, delay=delay)
        # print(impulse.shape, impulse[:, 0, 0].shape, step_func.shape)
        
        tdr = np.ones((time.shape[0], self.n_port), dtype=np.float64)
        for i in range(self.n_port):
            reflection_profile = np.convolve(impulse[:, i, i], step_func)
            
            n_convolve = len(reflection_profile)
            reflection_profile = np.real(reflection_profile[n_convolve//2-time.shape[0]//2: n_convolve//2+time.shape[0]//2]) #* 把合理的時間範圍取出來
            
            z0 = self.network.z0[0, i].real
            tdr[:,i] = z0*(1+reflection_profile)/(1-reflection_profile)
        
        
        #* Save TDR Information
        self._tdr_window = window
        self._tdr_rise_time = rise_time
        self._tdr_delay = delay
        self._tdr_time = time
        self._tdr = tdr
        
        return time, tdr
    
    
    # def _load_touchstone(self, touchstone_filepath: str):
    #!     ''' 
    #!     目前 skrf.Network 讀取資料已經足夠快了，暫時不需要自己寫讀取函數。
    #!     如果會用到，需要用到 `ma2cmplx` 、 `db2cmplx` 和 `ri2cmplx` 這幾個函數。
    #!     '''
        
    #     #$ Check the touchstone file extension
    #     file_extension = touchstone_filepath.split('.')[-1]
    #     if mo:=re.search(r's(\d+)p', file_extension):
    #         n_port = int(mo.groups()[0])
    #     else:
    #         raise ValueError('Invalid file extension: no port number.')
    #     #$ Read the touchstone file
    #     comment_lines: list[str] = []
    #     option_line: str = None
    #     data: list[str] = []        
    #     with open(touchstone_filepath, 'r') as file_io:
    #         for line in file_io.readlines():
    #             line = line.strip()
                
    #             #* Deal with comment lines 
    #             if line.startswith('!'):
    #                 comment_lines += [line]
                    
    #             #* Deal with option line
    #             elif line.startswith('#'):
    #                 if option_line is not None:
    #                     raise ValueError('Multiple option lines are found in the touchstone file.')
                    
    #                 option_line = line
                    
    #                 _, freq_unit, ntwk_type, data_format, __, z_ref = re.findall(r'\S+', line)
    #                 if __ != 'R':
    #                     raise ValueError(f'Invalid option line: {__}!=R ({line})')
                    
    #                 #* Check if the network type is S-parameter or not
    #                 if ntwk_type != 'S':
    #                     raise ValueError(f'Currently, only S-parameter is supported. Found {ntwk_type}-parameter.')
                    
    #                 #* Check if the data type is valid or not
    #                 if data_format.lower() not in ['ma', 'db', 'ri']:
    #                     raise ValueError(f'Invalid data type: {data_format}.')
                
    #             #* Deal with data
    #             else:
    #                 data += re.findall(r'\S+', line)
                    

    #     #$ Process parsed data
    #     #: 1. Parse the comment lines
    #     ansys_port_pattern = re.compile(r'!\s*Port\[(\d+)\] \= (\S+)')
    #     cadence_port_pattern = re.compile(r'!\s*([^\s\:]+\:\:[^\s\:]+)')
        
    #     comment_line: str = '\n'.join(comment_lines)
    #     if mo:=ansys_port_pattern.findall(comment_line):
    #         port_names = ansys_port_pattern.findall(comment_line)
    #     elif mo:=cadence_port_pattern.findall(comment_line):
    #         port_names = cadence_port_pattern.findall(comment_line)
    #     else:
    #         port_names = [f'Port_{i}' for i in range(n_port)]
        
    #     if len(port_names) != n_port:
    #         raise ValueError(f'Port number from file extension mismatch with the number of port names in the comment lines.')
        # pass
    
    
    # def __getattribute__(self, name: str) -> Any:
    #     if name in self.SUPPORTED_NETWORK_DATA_PROPERTIES and name != 'tdr':
    #         return getattr(self.network, name)
    #     else:
    #         return getattr(self, name)
        
    # def __getattr__(self, name: str) -> Any:
        
    #     return getattr(self.network, name, None) 
    

# t4 = datetime.now()

if '__main__' == __name__:
    
    # print(f'Loading takes {t2-t1}')
    # print(f'Function takes {t3-t2}')
    # print(f'Objects takes {t4-t3}')
    
    import matplotlib.pyplot as plt
    import time
    t1 = time.time()
    path  = r'd:\Users\szuhsien.feng\Desktop\TEMP\Realtek_RLE0758_proposal_design_051519-3_group1_model\RLE0758_190515_SerDes_Signal_TXRX_group1.s32p'
    path = r'D:/Users/szuhsien.feng/Desktop/TEMP/Test_DX/outputs/3_WBGA_Channel_240711_194006.s4p'
    path  =r'D:/Users/szuhsien.feng/Desktop/__MyRepositories/RT_Worklog/1.Software/1-1.DesignXplorer/4.Software/GUI/Test/QT-API/3_WBGA_Channel_240711_194006.s4p'
    
    ntwk_data = NetworkData(path)
    ntwk_data.single_to_mixmode()
    t2 = time.time()
    print(f'Loading takes {t2-t1}sec')
    
    t, tdr = ntwk_data.calculate_tdr()
    t3 = time.time()
    print(f'TDR takes {t3-t2}sec')
    # for i in range(tdr.shape[1]-2):
    #     # print(i)
    plt.plot(ntwk_data._tdr_time, ntwk_data._tdr[:, 0], ls='--')
    # plt.xlim(0, 2e-9)
    # plt.ylim(20, 60)
    # print(f'Loading take {t2-t1}sec')
    #! Test the NetworkData class
    #* Load

