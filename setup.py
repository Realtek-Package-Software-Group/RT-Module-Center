import os
import sys
import platform
import importlib.util

from setuptools import setup, find_packages
from setuptools.extension import Extension
from distutils.command.build_ext import build_ext

__MODULE_NAME__ = 'rt_ckt_api'
__VERSION__ = '0.0.1'
__AUTHOR__ = 'Jeff.Chou / Szuhsien.Feng'
__UPDATED__ = '2024.07.18'
__DESCRIPTION__ = f'API for circuit models (snp/spice) processing and check (updated at {__UPDATED__})'
__REQUIREMENT__ = ['numpy', 'numba', 'numexpr', 'skrf']

class CustomBuildExt(build_ext):
      
    user_options = build_ext.user_options + [('enable-opt', None, 'Enable optimization'), ]
    
    def initialize_options(self):
        super().initialize_options()
        self.enable_opt = False

    def finalize_options(self) -> None:
        super().finalize_options()
        
        if not self.extensions:
            return
        
        if platform.system() == 'Windows':
            if self.enable_opt:
                extra_compile_args = ['/O2']  # 開啟優化
            else:
                extra_compile_args = ['/Od']  # 關閉優化以提高Compile速度
        elif platform.system() == 'Linux':
            if self.enable_opt:
                extra_compile_args = ['-O3']  # 開啟優化
            else:
                extra_compile_args = ['-O0']  # 關閉優化以提高Compile速度
                
        for ext in self.extensions:
            ext.extra_compile_args = extra_compile_args


root_path = os.path.dirname(__file__)

# data files
hook_file_dir = os.path.join(os.path.dirname(__file__), 'rt_math_api', 'hook')

pyinstaller_spec = importlib.util.find_spec('PyInstaller')
if pyinstaller_spec is not None and pyinstaller_spec.submodule_search_locations:
    pyinstaller_path = pyinstaller_spec.submodule_search_locations[0]
    hooks_path = os.path.join(pyinstaller_path, 'hooks')
    if platform.system() == 'Windows':
        relative_hooks_path = hooks_path[hooks_path.index('Lib')::]  # ex: Lib\site-packages\PyInstaller\hooks
    elif platform.system() == 'Linux':
        relative_hooks_path = hooks_path[hooks_path.index('lib')::]  # ex: lib/python3.11/site-packages/PyInstaller/hooks
else:
    print('PyInstaller 模組未找到或無法定位hooks目錄')
    sys.exit(1)

data_files = [(relative_hooks_path, [os.path.join(hook_file_dir, hook_file) for hook_file in ['hook-rt_ckt_api.network.py', ]]), ]

setup(name=__MODULE_NAME__, \
      version=__VERSION__, \
      description=__DESCRIPTION__, \
      packages=find_packages(), \
      install_requires=__REQUIREMENT__, \
      data_files=data_files, \
      cmdclass={'build_ext': CustomBuildExt}, \
      author=__AUTHOR__, \
      )
#ext_modules=pyd_extensions

