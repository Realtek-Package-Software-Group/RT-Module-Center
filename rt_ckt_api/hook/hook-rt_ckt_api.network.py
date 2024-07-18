import os
import rt_nx_api.eda as eda
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = []
hiddenimports += collect_submodules('numba')
hiddenimports += collect_submodules('skrf')

datas = []
datas += [(os.path.join(os.path.dirname(eda.__file__), 'aedt_export_s_parameters.py'), os.path.join('rt_nx_api', '.'))]
datas += [(os.path.join(os.path.dirname(eda.__file__), 'aedt_export_bbs.py'), os.path.join('rt_nx_api', '.'))]
