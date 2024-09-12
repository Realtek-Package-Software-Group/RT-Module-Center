
import math
import numpy as np

def calculate_rectangular_cross_section_dc_inductance(width, thickness, length, mu_r=1):
    """
    Reference equation is (24) in the following paper:
    
    Z. Piatek, B. Baron, T. Szczegielniak, D. Kusiak and A. Pasierbek, 
    "Self inductance of long conductor of rectangularcross section", 
    Przeglad Elektrotechniczny Electr. Rev., vol. 88, no. 8, pp. 323-326, 2012.
    
    Benchmark:
    
    - case 1: width=1um; thickness=20um; length=100um; 
    
      calculate_rectangular_cross_section_dc_inductance(1e-6, 20e-6, 100e-6, 1) -> 55.04 pH
      Q3D -> 56.4 pH
      (Eq-Q3D)/Q3D -> -2.41%
      (Eq-Q3D) ->   -1.36 pH
    
    
    - case 2: width=5um; thickness=20um; length=100um
    
      calculate_rectangular_cross_section_dc_inductance(5e-6, 20e-6, 100e-6, 1) -> 51.54 pH
      Q3D -> 52.94 pH
      (Eq-Q3D)/Q3D -> -2.644%
      (Eq-Q3D) ->   -1.4 pH
      
    - case 3: width=10um; thickness=20um; length=100um
    
      calculate_rectangular_cross_section_dc_inductance(10e-6, 20e-6, 100e-6, 1) -> 48 pH
      Q3D -> 49.48 nH
      (Eq-Q3D)/Q3D -> -3.00%  
        (Eq-Q3D) ->   -1.48 pH
      
    - case 4: width=20um; thickness=20um; length=20um
    
      calculate_rectangular_cross_section_dc_inductance(20e-6, 20e-6, 20e-6, 1) -> 1.99 pH
      Q3D -> 3.77 pH    
      (Eq-Q3D)/Q3D -> 47%  
      (Eq-Q3D) ->   1.78 pH

    - case 5: width=20um; thickness=20um; length=50um
    
      calculate_rectangular_cross_section_dc_inductance(20e-6, 20e-6, 50e-6, 1) -> 14.1 pH
      Q3D -> 16 pH  
      (Eq-Q3D)/Q3D -> -12.5%   
        (Eq-Q3D) ->   -1.9 pH
      
    - case 6: width=20um; thickness=20um; length=100um
    
      calculate_rectangular_cross_section_dc_inductance(20e-6, 20e-6, 100e-6, 1) -> 42.2 pH
      Q3D -> 44 pH      
      (Eq-Q3D)/Q3D -> -3.64%   
        (Eq-Q3D) ->   -1.8 pH

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
    
    
