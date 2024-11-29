api_name = 'rt_module_utility'

import os

root_path = os.path.dirname(__file__)
setup_py_path = os.path.join(root_path, 'setup.py')
os.system(f'python {setup_py_path} build_ext --inplace')

try:
    os.system(f'pip uninstall -y {api_name}')
except:
    pass
os.system(f'pip install --upgrade {root_path}')