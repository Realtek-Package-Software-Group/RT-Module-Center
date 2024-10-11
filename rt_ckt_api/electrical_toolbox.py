
import math
import numpy as np

def calculate_rectangular_cross_section_dc_self_inductance(width: int|float, thickness: int|float, length: int|float, mu_r:int|float = 1.0) -> int|float:

    """
    Reference equation is the one of (24) in the following paper:
    
    Z. Piatek, B. Baron, T. Szczegielniak, D. Kusiak and A. Pasierbek, 
    "Self inductance of long conductor of rectangularcross section", 
    Przeglad Elektrotechniczny Electr. Rev., vol. 88, no. 8, pp. 323-326, 2012.
    
    Benchmark:

    | Case | Condition               | Eq. (pH) | Q3D (pH) | (Eq-Q3D)/Q3D (%) | (Eq-Q3D) (pH) |
    |------|-------------------------|----------|----------|------------------|---------------|
    | 1    | w=1um, t=20um, l=100um  | 55.04    | 56.4     | -2.41            | -1.36         |
    | 2    | w=5um, t=20um, l=100um  | 51.54    | 52.94    | -2.644           | -1.4          |
    | 3    | w=10um, t=20um, l=100um | 48.00    | 49.48    | -3.00            | -1.48         |
    | 4    | w=20um, t=20um, l=20um  | 1.99     | 3.77     | 47               | 1.78          |
    | 5    | w=20um, t=20um, l=50um  | 14.1     | 16       | -12.5            | -1.9          |
    | 6    | w=20um, t=20um, l=100um | 42.2     | 44       | -3.64            | -1.8          |
    
    """
    
    assert (width >= 0 and thickness >= 0 and length >= 0), "Width, thickness and length must be positive"
    
    a = width
    b = thickness
    l = length
    
    
    mu0 = 4e-7 * math.pi 
    value = mu_r*mu0/2/math.pi * l * (
        math.log(2*l/(a+b)) 
        + 13/12
        - 2/3 * (b/a * math.atan(a/b) + a/b * math.atan(b/a))
        + 0.5 * math.log(1 + 2*a/b/(1+(a/b)**2))
        + 1/12 * ((a/b)**2 * math.log(1+(b/a)**2) + (b/a)**2 * math.log(1+(a/b)**2))
        )
    
    assert value >= 0, "Inductance must be positive"
    
    return value
    
def calculate_rectangular_cross_section_dc_mutual_inductance(width_1: int|float, thickness_1:int|float, length_1: int|float, corner1: np.ndarray|tuple|list, 
                                                             width_2: int|float, thickness_2:int|float, length_2: int|float, corner2: np.ndarray|tuple|list, 
                                                             mu_r: float|int = 1.0):
    """
    Reference equation is the one of (5) and (7) in the following paper:
    
    Z. Piatek, B. Baron, T. Szczegielniak, D. Kusiak and A. Pasierbek, 
    "Exact closed form formula for mutual inductance of conductors of rectangular cross section", 
    Przeglad Elektrotechniczny, vol. 89, no. 3a, pp. 61-64, 2013.    
    
    Benchmark:
    
    | Case | Condition                                                                         | Eq. (pH) | Q3D (pH) |
    |------|-----------------------------------------------------------------------------------|----------|----------|
    | 1    | a1=20um; a2=10um; b1=1um; b2=1um; l1=100um; l2=100um; Spacing=75um                | 10.28    | 10.29    |
    | 2    | a1=20um; a2=10um; b1=1um; b2=1um; l1=100um; l2=150um; Spacing=75um                | 14.58    | 14.58    |
    | 3    | a1=20um; a2=10um; b1=1um; b2=1um; l1=100um; l2=150um; Spacing=15um                | 31.04    | 31.04    |
    | 4    | a1=20um; a2=10um; b1=1um; b2=15um; l1=100um; l2=100um; Spacing=15um               | 30.35    | 30.35    |
    | 5    | a1=20um; a2=10um; b1=1um; b2=15um; l1=50um; l2=100um; Spacing=15um                | 14.15    | 14.15    |
    | 6    | a1=20um; a2=10um; b1=1um; b2=15um; l1=50um; l2=100um; Spacing=15um; offset_x=30um | 10.88    | 10.88    |
    
    Example:
    
    
    """
    def F(x, y, z):
        
        x = round(x, 10)
        y = round(y, 10)
        z = round(z, 10)        
        r = (x**2 + y**2 + z**2)**0.5
        
        term1 = 6/5 * (x**4 + y**4 + z**4 - 3* (x**2*y**2 + y**2*z**2 + z**2*x**2)) * r
        if x == 0 or y == 0 or z == 0:
            term2 = 0
        else:
            term2 = -12 * x*y*z * (z**2 * math.atan(x*y/(z*r)) + y**2 * math.atan(x*z/(y*r)) + x**2 * math.atan(y*z/(x*r)))
        
        if y == 0 and z == 0:
            term3 = 0
        else:
            term3 = -3 * x * (y**4 - 6*y**2*z**2 + z**4) * math.log(x + r)
        
        if x == 0 and z == 0:
            term4 = 0
        else:   
            term4 = -3 * y * (x**4 - 6*x**2*z**2 + z**4) * math.log(y + r)
            
        if x == 0 and y == 0:
            term5 = 0
        else:
            term5 = -3 * z * (x**4 - 6*x**2*y**2 + y**4) * math.log(z + r)
            
        res = 72**-1 * (term1 + term2 + term3 + term4 + term5)
            
        
        assert isinstance(res, complex) == False, "Result must be real"
        return res

    s1, s5, s9 = corner1
    s2, s6, s10 = corner1 + np.array((width_1, thickness_1, length_1))
    s3, s7, s11 = corner2
    s4, s8, s12 = corner2 + np.array((width_2, thickness_2, length_2))

    mu0 = 4e-7 * math.pi 
    factor = mu0 / 4/ math.pi / (width_1*width_2*thickness_1*thickness_2)
    
    value = 0
    for i in range(1, 5):
        if i == 1:
            p = s1-s4
        elif i == 2:
            p = s1-s3
        elif i == 3:
            p = s2-s3
        else:
            p = s2-s4        
        
        for j in range(1, 5):
            if j == 1:
                q = s5-s8
            elif j == 2:
                q = s5-s7
            elif j == 3:
                q = s6-s7
            else:
                q = s6-s8
                
            for k in range(1, 5):
                if k == 1:
                    r = s9-s12
                elif k == 2:
                    r = s9-s11
                elif k == 3:
                    r = s10-s11
                else:
                    r = s10-s12
                
                value += F(p, q, r) * (-1)**(i+j+k+1)

    # assert value >= 0, f"Mutual Inductance must be positive: {value}"
    return factor*value


if __name__ == "__main__":
    width_1 = 1810e-6/3
    thickness_1 = 1.12e-6
    length_1 = 2210e-6  
    corner1 = np.array((0,0,0))
    print(calculate_rectangular_cross_section_dc_self_inductance(width_1, thickness_1, length_1))
    
    width_1 = 1810e-6
    thickness_1 = 1.12e-6
    length_1 = 2210e-6  
    corner1 = np.array((0,0,0))
    print(calculate_rectangular_cross_section_dc_self_inductance(width_1, thickness_1, length_1))
    
    width_1 = 137e-6
    thickness_1 = 1.12e-6
    length_1 = 1430e-6  
    corner1 = np.array((0,0,0))
    print(calculate_rectangular_cross_section_dc_self_inductance(width_1, thickness_1, length_1))
    
    print(calculate_rectangular_cross_section_dc_self_inductance(102e-6, 1.12e-6, 403e-6) + calculate_rectangular_cross_section_dc_self_inductance(181e-6, 1.12e-6, 881e-6))
    
    
    
    
    
    width_2 = 10e-6
    thickness_2 = 15e-6
    length_2 = 150e-6
    spacing = 15e-6
    
    offset_y = 30e-6
    corner2 = np.array((width_1+spacing,offset_y,0))
    
    print(calculate_rectangular_cross_section_dc_self_inductance(width_1, thickness_1, length_1))
    # print(calculate_rectangular_cross_section_dc_mutual_inductance(width_1, thickness_1, length_1, corner1, width_2, thickness_2, length_2, corner2))
    