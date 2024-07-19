import fnmatch
import os
import timeit
import sys

# 考慮沒安裝必要套件的狀況
import importlib
import subprocess
import sys
not_install_packages = list()
for package_name in ['setuptools', 'wheel']:
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError:
        not_install_packages.append(package_name)
        
for package_name in not_install_packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}, use NX cmd instead...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", "https://pypi/simple", "--trusted-host", "pypi", package_name])


# 定義要刪除的檔案類型
patterns = ['*.pyd', '*.c', '*.so', '*.dll']

# 遍歷目錄並刪除匹配的檔案
root_path = os.path.dirname(__file__)
for root, dirs, files in os.walk(os.path.join(root_path, 'rt_ckt_api')):
    for pattern in patterns:
        for filename in fnmatch.filter(files, pattern):
            os.remove(os.path.join(root, filename))

setup_py_path = os.path.join(root_path, 'setup.py')

t1 = timeit.default_timer()
os.system(f'python {setup_py_path} build_ext --inplace')
t2 = timeit.default_timer()
print('root_path: ', root_path)
os.system(f'pip install --upgrade {root_path}')

print(f'{t2-t1}sec')