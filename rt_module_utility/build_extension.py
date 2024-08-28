
import platform
import os
from distutils.command.build_ext import build_ext

class CustomBuildExt(build_ext):
      
    user_options = build_ext.user_options + [('enable-opt', None, 'Enable optimization'), ]
    
    def initialize_options(self):
        super().initialize_options()
        self.enable_opt = False

    def build_extensions(self):
            self.parallel = os.cpu_count()//2  # 自动使用 CPU 核心数
            super().build_extensions()

    def finalize_options(self) -> None:
        super().finalize_options()
        
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


__all__ = ['CustomBuildExt', ]