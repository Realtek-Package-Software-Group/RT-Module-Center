from setuptools import setup

__MODULE_NAME__ = 'rt_module_utility'
__VERSION__ = '0.0.1'
__AUTHOR__ = 'Jeff.Chou'
__UPDATED__ = '2024.08.28'
__DESCRIPTION__ = f'Utilities for realtek module development (updated at {__UPDATED__})'
__REQUIREMENT__ = []


setup(name=__MODULE_NAME__, \
      version=__VERSION__, \
      description=__DESCRIPTION__, \
      install_requires=__REQUIREMENT__, \
      author=__AUTHOR__, \
      )