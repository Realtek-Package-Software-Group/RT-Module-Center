import numpy as np
import skrf as rf
import numba as nb
import numpy as np
from typing import Optional, Callable, Any, Sequence
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from rt_math_api.utility import ExpressionValue, UNIT_TO_VALUE
from rt_nx_api.eda import CMDThread
import re
import os
import platform
import threading
import shutil
import subprocess

try:
    import ctypes
    import msvcrt
except:
    pass

#%%


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
                     f'-prof {copied_filepath.parent/"log.txt"}',
                     f'-cccontinuation 0 ',
                     f'-ccinterp 1',
                     f'-ccintegration 2',
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
    
    causality_infomation['data'] = rf.Network(str(copied_filepath))
    return causality_infomation


#%%


path = r'D:/Users/szuhsien.feng/Desktop/TEMP/TestCausality/RL7025_BGAc_AFE_Q3DSNP_20240708.s103p'
path = r'D:/Users/szuhsien.feng/Desktop/TEMP/TestCausality/my_sparam_1p0.s4p'
path= r'D:/Users/szuhsien.feng/Desktop/TEMP/TestCausality/3_WBGA_Channel_240704_153134.s4p'
causality_infomation = check_touchstone_by_genequiv(path, causality_tolerance=0.00001)


#%%
ntwk = causality_infomation['data']
recon_ntwk = causality_infomation['reconstructed_data']
disc_error = causality_infomation['discretization_error']
trunc_ntwk = causality_infomation['truncation_error'].s[:,0,0]

n_freq = ntwk.f.shape[0]
n_port = ntwk.nports

causality_value = np.zeros_like(ntwk.s, dtype=np.float64)
upper_bound_s = np.zeros_like(ntwk.s,)
lower_bound_s = np.zeros_like(ntwk.s,)
tol_err = 0.01

for k in range(n_freq):
    for i, j in ntwk.port_tuples:
        s = recon_ntwk.s[k, i, j]
        upper_bound_s[k, i, j] = (abs(s)  + disc_error.s[k, i, j] + trunc_ntwk[k]) / abs(s) * s
        lower_bound_s[k, i, j] = (abs(s)  - disc_error.s[k, i, j] - trunc_ntwk[k]) / abs(s) * s
        
        # causality_value[k, i, j] = abs(ntwk.s[k, i, j] - recon_ntwk.s[k, i, j]) - disc_error.s[k, i, j].real - trunc_ntwk[k] -  tol_err

#%%
# plt.plot(ntwk.f, db(abs(recon_ntwk.s-ntwk.s)[:,0,0]), label='reconstruct error DB')
# plt.plot(ntwk.f, db((disc_error.s)[:,0,0]+trunc_ntwk), label='error bound DB')
# plt.plot(ntwk.f, db(trunc_ntwk), label='error bound DB2')
# plt.plot(ntwk.f, db(abs(recon_ntwk.s-ntwk.s)[:,0,0])-db((disc_error.s)[:,0,0]+trunc_ntwk), label='reconstruct error DB')


plt.plot(ntwk.f, abs(recon_ntwk.s-ntwk.s)[:,0,0], label='reconstruct error')
plt.plot(ntwk.f, abs(disc_error.s)[:,0,0]+trunc_ntwk, label='error bound')
# plt.plot(ntwk.f, [tol_err]*ntwk.f.shape[0])
plt.yscale('log')
plt.legend()




#%%


db = lambda x : np.log10(x) * 20
my_type = 'real'

if my_type == 'db':
    myfunc = db
    mylim = []
elif my_type == 'real':
    myfunc = np.real
    mylim = (-0.6, -0.3)
else:
    myfunc = np.imag
    mylim = (-0.1, 0.15)



plt.figure(dpi=500)
plt.plot(myfunc(ntwk.s[:,0,0]), label='RawData')
plt.plot(myfunc(upper_bound_s[:,0,0]), label='UpperBound', ls='--')
plt.plot(myfunc(lower_bound_s[:,0,0]), label='LowerBound',ls='--')

# if mylim:
#     plt.ylim(*mylim)

plt.legend()






