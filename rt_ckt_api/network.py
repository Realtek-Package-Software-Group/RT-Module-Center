import numpy as np
import skrf as rf
import numba as nb
from typing import Optional, Callable, Any, Sequence

#%% Functions
@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(nb.float64, nb.float64, nb.int64, nb.float64, nb.float64))
def generate_step_function(start_time, end_time, num_points, rise_time=50e-12, delay=1e-9):
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


#%% Objects


class NetworkData():
    
    #* Default parameters
    NETWORK_TYPE: list[str] = ['s', 'y', 'z', ]
    DATA_TYPE: list[str] = ['mag', 'db', 'deg', 're', 'im']    
    
    
    n_fitting = 2000 #* Used for vector fitting
    
    raw_network: Optional[rf.Network] = None
    fitting_network: Optional[rf.Network] = None
    

    
    
    
    def __init__(self):
        pass
    
    def _load_s_parameters(self):
        pass




