import numpy as np
import skrf as rf
import numba as nb
import numpy as np
from typing import Optional, Callable, Any, Sequence
from pathlib import Path
from datetime import datetime


from rt_math_api.utility import ExpressionValue, UNIT_TO_VALUE
from rt_nx_api.eda import CMDThread
import re
import os
import platform
import threading
import shutil

#%% Math Functions

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


def s2z():
    '''To be continued'''
    pass

def s2y():
    '''To be continued'''
    pass



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

class WindowsCMDThread(threading.Thread):
    '''
    Copy CustomThread in eda.py to do similar job in WindowS
    
    Notes: when Windows return output, \r is also used.
    '''
    _thread_id = None
    stdout_log_text = ''
    stderr_log_text = ''
    cmd_process = None
    sub_process_collecter = None # * Use for killing sub-process
    stop_flag = False
    
    def __init__(self, cmd ):
        
        self.cmd = cmd
        self.custom_killed = False
        
        super().__init__(daemon=False)
        
    def run(self):
        
        self.cmd_process = subprocess.Popen(self.cmd, shell=True, text=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        #* Start to collect sub-process pid
        # self.sub_process_collecter = SubProcessThread(self.cmd_process.pid)
        # self.sub_process_collecter.start()
        # print('This is ', self.cmd_process.pid)
        self.process_output()
        
        #* Make sure the process is stop properly
        # while True:
        #     if self.cmd_process.poll() is not None:
        #         self.sub_process_collecter.is_closed = True # * Let sub_process_collecter kill sub-process
        #         break
        #     time.sleep(1)


    def set_non_blocking_pipe(self, fd):
        PIPE_NOWAIT = ctypes.wintypes.DWORD(0x00000001)
        handle = msvcrt.get_osfhandle(fd)
        ctypes.windll.kernel32.SetNamedPipeHandleState(handle, ctypes.byref(PIPE_NOWAIT), None, None)

    def is_data_available(self, fd):
        handle = msvcrt.get_osfhandle(fd)
        available = ctypes.wintypes.DWORD()
        ctypes.windll.kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None)
        return available.value > 0

    def read_non_blocking(self, fd):
        handle = msvcrt.get_osfhandle(fd)
        available = ctypes.wintypes.DWORD()
        ctypes.windll.kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None)
        if available.value > 0:
            return os.read(fd, available.value)
        return b''

    def process_output(self):
        
        if not self.cmd_process:
            return
        
        
        stdout_partial = ''
        stderr_partial = ''       
        
        assert self.cmd_process.stdout is not None
        assert self.cmd_process.stderr is not None
        stdout_fileno = self.cmd_process.stdout.fileno()
        stderr_fileno = self.cmd_process.stderr.fileno()         
        while True:
            if self.is_data_available(stdout_fileno):
                stdout_partial += self.read_non_blocking(stdout_fileno).decode('utf-8', errors='ignore')
                lines = stdout_partial.split('\n')
                stdout_partial = lines.pop()  # 保留最後一個未完成的行
                for line in lines:
                    self.stdout_log_text += f'{line.strip()}\n'
                    print(f"[STDOUT] {line}")

            if self.is_data_available(self.cmd_process.stderr.fileno()):
                stderr_partial += self.read_non_blocking(stderr_fileno).decode('utf-8', errors='ignore')
                lines = stderr_partial.split('\n')
                stderr_partial = lines.pop()  # 保留最後一個未完成的行
                for line in lines:
                    self.stderr_log_text += f'{line.strip()}\n'
                    print(f"[STDERR] {line}")

            if self.cmd_process.poll() is not None:
                break

        # 處理最後一部分緩衝區
        if stdout_partial:
            self.stderr_log_text += f'{stdout_partial.strip()}\n'
            print(f"[STDOUT] {stdout_partial}")
        if stderr_partial:
            self.stderr_log_text += f'{stderr_partial.strip()}\n'
            print(f"[STDERR] {stderr_partial}")

        # close stream
        self.cmd_process.wait() # wait for sub-process to finish
        self.cmd_process.stdout.close()
        self.cmd_process.stderr.close()
        
    @property
    def thread_id(self):
        if self._thread_id is None:
            self._thread_id = ctypes.c_long(self.ident)
        return self._thread_id
    
    def raise_exception(self):
        if self.daemon:
            print('Can not raise an exception to a daemon thread!!')
        else:
            result = ctypes.pythonapi.PyThreadState_SetAsyncExc(self.thread_id, ctypes.py_object(SystemExit))
            print(f'thread:raise_exception', result)
            if result == 0:
                print(f'invalid thread id:{self.thread_id}')
                raise ValueError(f'invalid thread id:{self.thread_id}')
            elif result == 1:
                print(f'thread {self.thread_id} terminated')
            elif result > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(self.thread_id, None)
                print(f"thread {self.thread_id}:PyThreadState_SetAsyncExc failed")
                raise SystemError(f"thread {self.thread_id}:PyThreadState_SetAsyncExc failed")
            
    def custom_kill(self):
        self.custom_killed = True
        self.raise_exception()
        

def execute_cmd_thread(cmd: str) -> threading.Thread:
    
    if platform.system() == 'Windows':
        thread = WindowsCMDThread(cmd)
    elif platform.system() == 'Linux':
        thread = CMDThread(cmd)
    else:
        raise ValueError('Unsupported Platform')
    
    thread.start()
    return thread

def check_touchstone_by_genequiv(touchstone_filepath: str, check_passivity: bool = True, check_causality: bool = True, causality_tolerance: float = 0.01, cpu_count: int = -1):
    
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
    copied_filepath: Path = Path(filepath.parent / f'.CheckTouchstone_{serial_number}' / filepath.name)
    if not copied_filepath.parent.exists():
        copied_filepath.parent.mkdir()
    shutil.copy(str(filepath), str(copied_filepath))
    
    #* Prepare the command flags
    cmd_list: list[str] = [f'"{genequiv_exepath}"']
    if check_passivity:
        cmd_list += ['-checkpassivity'] #: Check passivity
    if check_causality:
        cpu_count = os.cpu_count()//2 if cpu_count < 0 else cpu_count # type:ignore
        cmd_list += ['-checkcausality', #: Check causality
                     '-causality_plots', #: Generate reconstructed touchstone and error/truncation bound touchstone
                     f'-causality_tol {causality_tolerance}', #: Causality tolerance
                     f'-mp {cpu_count//2}', #: Multi-processing number
                     f'-i {copied_filepath}' #: Input touchstone filepath
                     ] 
    
    #* Need to  change director
    if platform.system() == 'Windows':
        command = f'cd "{copied_filepath.parent}"&' + ' '.join(cmd_list)
    else:
        command = f'cd "{copied_filepath.parent}";' + ' '.join(cmd_list)
    
    
    print(command, '\n')
    
    thread = execute_cmd_thread(command)
    thread.join()
    
    #: Just for checking the result
    # result_filepaths: list[str] = []
    # for name in copied_filepath.parent: # type:ignore
    #     if 'DiscErrBnd' in name or 'ReconsData' in name or 'TruncErrBnd' in name:
    #         result_filepaths += [name]
            
    causality_infomation: dict[str, rf.Network] = {}
    for name in os.listdir(copied_filepath.parent): # type:ignore
        if 'DiscErrBnd' in name:
            causality_infomation['discretization_error'] = rf.Network(str(copied_filepath.parent / name))
        elif 'ReconsData' in name:
            causality_infomation['reconstructed_data'] = rf.Network(str(copied_filepath.parent / name))
        elif 'TruncErrBnd' in name:
            causality_infomation['truncation_error'] = rf.Network(str(copied_filepath.parent / name))
        
    if set(causality_infomation.keys()) != {'discretization_error', 'reconstructed_data', 'truncation_error'}:
        raise ValueError('Causality check failed.')
    
    
    return causality_infomation
    
    


#%% Objects
class NetworkData():
    
    #* Default parameters
    SUPPORTED_NETWORK_DATA: dict[str, list[str]] = {
        's': ['mag', 'db', 'deg', 're', 'im'],
        # 'y': ['mag', 'db', 'deg', 're', 'im'],
        # 'z': ['mag', 'db', 'deg', 're', 'im'],
        # 'tdr': [''],
        }
    SUPPORTED_NETWORK_DATA_PROPERTIES: list[str] = [f'{ntwk_type}_{data_type}' for ntwk_type, data_type_list in SUPPORTED_NETWORK_DATA.items() for data_type in data_type_list]
    
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
    _passivity_matrix: np.ndarray = None
    _causality_matrix: np.ndarray = None
    
    def __init__(self, touchstone_filepath: str = '', s_parameter: Optional[np.ndarray] = None, freq: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None, ):
        
        #* Load the raw network data
        if touchstone_filepath:
            ts = rf.Touchstone(touchstone_filepath)
            f: np.ndarray = ts.f
            s: np.ndarray = ts.s
            n_freq: int = s.shape[0]
            n_port: int = s.shape[1]
            reference_impedance = ts.resistance # type:ignore
            # print('reference_impedance', reference_impedance)
            #* Check if `reference_impedance` is wrongly parsed 
            expected_z0 = np.tile(reference_impedance, (n_freq, n_port))  # type:ignore
            if not (ts.z0 == expected_z0).all():
                print(f'Warning: touchstone "{touchstone_filepath}" seems parse wrong reference impedance. Corrected to {reference_impedance}.')
            
            self.reference_impedance = reference_impedance
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
            if skrf_ntwk_prop is None:
                raise ValueError(f'Property {attr} is not supported by skrf.Network.')
            setattr(self, attr, skrf_ntwk_prop)

    @property
    def passivity_matrix(self):
        if self._passivity_matrix is None:
            self.check_passivity()
        return self._passivity_matrix

    def single_to_mixmode(self, ):
        '''Convert the single-ended network to mixed-mode network.'''
        if self.n_port % 2 == 1:
            raise ValueError('Currently, even port number is required for mixed-mode network.')
        

        n_channel = self.n_port // 2
        self.network.se2gmm(p=n_channel)
        
        if self.fitted_network:
            self.fitted_network.se2gmm(p=n_channel)
            
        self._dynamic_load_network_parameter(self.network) #* Reload the network parameters

    def check_passivity(self) -> list[tuple[int, str, float]]:
        passivity_matrix = calculate_passivity_matrix(self.s)

        passivity_reports: list = []
        for i in range(self.n_freq):
            for j in range(self.n_port):
                if passivity_matrix[i, j] > 1:
                    passivity_reports += [(i, str(ExpressionValue(self.f[i], unit='GHz')), passivity_matrix[i, j])]
        
        self._passivity_matrix = passivity_matrix
        return passivity_reports
    
    def check_causality(self):
        pass
    
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
    #     ''' 
    #     目前 skrf.Network 讀取資料已經足夠快了，暫時不需要自己寫讀取函數。
    #     如果會用到，需要用到 `ma2cmplx` 、 `db2cmplx` 和 `ri2cmplx` 這幾個函數。
    #     '''
        
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
    



if '__main__' == __name__:
    
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

