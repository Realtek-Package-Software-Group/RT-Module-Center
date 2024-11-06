from collections import defaultdict

class NamedTuple:
    """
    這是一個物件導向實現的Tuple結構, 支援傳統的tuple操作, 也可以直接用屬性名稱來獲取值
    """
    def __init__(self, *args, **kwargs):  # we can only use keyword to assign item
        
        if args:
            raise TypeError(f'Positional argument is not allowed for {self.__class__.__name__}')
        
        for key, value in kwargs.items():  # python 3.7之後, dict預設就是有序的, 因此不用擔心順序不對
            super().__setattr__(key, value)

        super().__setattr__('_attributes', list(kwargs.keys()))

    def __setattr__(self, key, value):  # to ensure the immutable feature of tuple
        raise AttributeError(f"Attribute '{key}' is immutable and cannot be modified")
    
    def __setitem__(self, index, value):  # to ensure the immutable feature of tuple
        raise TypeError(f"{self.__class__.__name__} object does not support item assignment")
        
    def __getitem__(self, index):
        
        if index.__class__ != int:
            raise TypeError("Index must be an integer")
        
        try:
            return getattr(self, self._attributes[index])
        except IndexError:
            raise IndexError("Index out of range")

    def __iter__(self):  # for unpack, ex: x, y = center_xy
        
        for attr in self._attributes:
            yield getattr(self, attr)

    def __len__(self):  # for len()

        return len(self._attributes)

    def __repr__(self):

        attrs = ", ".join(f"{key}={getattr(self, key)}" for key in self._attributes)
        return f"{self.__class__.__name__}({attrs})"
    
    def __lt__(self, other):  # for sorted()
        
        if other.__class__ == self.__class__:
            return tuple([attr for attr in self]) < tuple([attr for attr in other])
        elif other.__class__ == tuple:
            return tuple([attr for attr in self]) < other
        else:
            raise TypeError(f"'<' not supported between instances of '{self.__class__.__name__}' and '{other.__class__.__name__}'")

    def __eq__(self, other):  # for == 和 in 檢查
        
        if other.__class__ == self.__class__ and other._attributes == self._attributes:  # 考慮屬性名稱差異
            
            return all(getattr(self, attr) == getattr(other, attr) for attr in self._attributes)

        else:
            return False
        
        
    def __hash__(self):  # for set 和 dict-key 使用 (考慮屬性名稱差異)

        return hash(tuple((attr, getattr(self, attr)) for attr in self._attributes))
    
    def __bool__(self):
        
        if self._attributes:
            return True
        else:
            return False
            
        
    

class AutoDict:
    """
    這是一個可以自動生成多階 dict/list的資料結構
    """

    def __init__(self):
        self.data = {}  # internal-data-structure (can be dict or list)

    def __getitem__(self, key):

        if self.data.__class__ == dict:
            
            if key not in self.data:
                self.data[key] = self.__class__()  # 自動產生dict
            return self.data[key]
        
        elif self.data.__class__ == list:
            
            if key.__class__ != int:
                raise TypeError("Index must be an integer")
    
            try:
                return self.data[key]
            except IndexError:
                raise IndexError("Index out of range")
        
    def __setitem__(self, key, value):
        
        self.data[key] = value

    def __getattr__(self, name):

        if name == 'append' and self.data.__class__ == dict:
            
            if self.data:  # with existed value
                raise AttributeError(f"Can not use '{name}' method with non-empty dict: {self.data}")
            else:
                self.data = []  # 自動產生list
        
        try:
            return getattr(self.data, name)
        except AttributeError: 
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self):
        return repr(self.data)


if __name__ == '__main__':
    
    # NamedTuple Demo
    a = NamedTuple(x=1, y=2)
    
    print('屬性取值', a.x, a.y)
    
    x, y = a
    print('unpack', x, y)
    
    print('index取值', a[0], a[1])
    
    print(a)
    
    # AutoDict Demo
    a = AutoDict()
    a['L1']['L2'] = '12'
    a['L3'].append('3')
    a['L3'].append('4')

    print(a)
    print(a['L3'][0], a['L3'][1])
    a['L3'].extend(['5', '6'])
    print(a['L3'])
    
    a['L1'].update({'L22': '33'})
    print(a)

    b = [NamedTuple(x=3, y=2), NamedTuple(x=2, y=3)]
    print(b)
    b = sorted(b)
    print(b)
    
    print(NamedTuple(x=3, y=2) in b)
    print(NamedTuple(z=3, y=2) in b)
    c = dict()
    c[NamedTuple(x=3, y=2)] = '222'
    print(c)
    
    d = NamedTuple()
    if d:
        print('get tuple')