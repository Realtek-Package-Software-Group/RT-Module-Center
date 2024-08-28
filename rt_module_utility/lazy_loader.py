
from importlib import import_module

class LazyLoader:
    _module = None
    _member_name = None  # 通常是類/函數/常數等非模組的物件
    _local_name = None
    _parent_module_global = None

    def __init__(self, import_name: str, local_name: str, parent_module_globals: dict):

        self._import_name = import_name  # 導入對象的完整名稱 ex: numpy or PyQt6.QtCore or pandas.DataFrame
        self._local_name = local_name  # 實際使用的變量 ex: np, pd, tf
        self._parent_module_globals = parent_module_globals  # LazyLoader實例化對應的命名空間 (用於後續導入模組的時候更新)

    @property
    def module(self):
        if self._module is None:  # 避免重複導入
            try:

                self._module = import_module(self._import_name)  # 導入模組 (比如: numpy, PyQt6.QtCore)

            except ModuleNotFoundError:  # 導入類/函數/常數等非模組的物件, 會引發這類錯誤 (比如import numpy.array or import math.pi)

                if '.' in self._import_name:  # 從import_name中區分模組與非模組的名稱
                    module_name, member_name = self._import_name.rsplit('.', 1)  # ex: 'a.b.c.d'.rsplit('.', 1) --> ['a.b.c', 'd']
                    self._module = import_module(module_name)  # ex: math.pi的math
                    self._member_name = member_name  # ex: math.pi的pi
            
            self.__dict__.update(self._module.__dict__)  # 將實際模塊的屬性和方法更新到這個LazyLoader類的__dict__中, 避免每次存取屬性都要調用__getattr__ (而是直接從self.__dict__中獲取)
                                                         # 主要用來優化性能 

            # 更新parent_module_globals
            if self._member_name is not None:
                self._parent_module_globals[self._local_name] = getattr(self._module, self._member_name)
            else:
                self._parent_module_globals[self._local_name] = self._module

        return self._module

    @property
    def member_name(self) -> str | None:
        return self._memeber_name

    @property
    def member(self):
        if self.module is None:
            return
        if self._member_name is not None:
            return getattr(self.module, self._member_name)
        else:
            return self.module

    def __getattr__(self, item):
        return getattr(self.member, item)
    
    def __dir__(self):
        return dir(self.member)

    def __call__(self, *args, **kwargs):

        # 如果導入的是函數類的
        if callable(self.member):
            return self.member(*args, **kwargs)

        raise TypeError(f"{self._import_name} is not callable")


__all__ = ['LazyLoader', ]

if __name__ == '__main__':

    np = LazyLoader(import_name="numpy", local_name="np", parent_module_globals=globals())

    print(np)  # <__main__.LazyLoader object at ...>
    result = np.array([1, 2, 3])  # 這裡觸發會__getattr__
    print(np)  # <module 'numpy' from 'C:\\RTAutoSIM_env_py312\\Lib\\site-packages\\numpy\\__init__.py'> --> np已被更新為原始模組對象, 不再是LazyLoader
    print(result)