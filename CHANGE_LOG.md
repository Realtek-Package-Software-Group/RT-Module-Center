<!--Author:jeff.chou, Host:https://wiki.realtek.com/rest/api, SpaceKey:PKGSW, PageID:539913360, GitHub:https://github.com/Realtek-Package-Software-Group/RT-Math-API/blob/main/CHANGE_LOG.md-->

# 2024.06


## 🎉 New Features

<h3><code>utility</code>  module</h3>

This module is designed for generic functions and classes that are commonly used in various applications.  
Currently, the most important objects are responsible for handling numerical string.  
Two major objects are included in this module so far:

- `ExpressionValue`: A class for handling numerical values with units and precision.  

<div style="margin-left: 40px">

```python
print(ExpressionValue('100um') - '10um') # Output: 90.00um
print(ExpressionValue('100um') + '10um') # Output: 100.00um
print(ExpressionValue('100um') * 5.0) # Output: 500.00um
print(ExpressionValue('100um') / '50um') # Output: 2.00
print('50um' / ExpressionValue('100um')) # Output: 0.50
```

</div>

- `ExpressionEvaluator`: A class for evaluating and formatting numerical expressions.  

<div style="margin-left: 40px">

```python
print(ExpressionEvaluator.evaluate('10in', 'mil')) # Output: 10000.0
print(ExpressionEvaluator.format_string('10in', 'mil', 2)) # Output: 10000.00mil
print(ExpressionEvaluator.evaluate('123.456mm', 'um')) # Output: 123456.0
print(ExpressionEvaluator.format_string('123.456mm', 'um')) # Output: 123456.00um
print(ExpressionEvaluator.format_string('123.456mm', 'mm')) # Output: 123.45600mm
```

</div>


# 2024.04

## 🎉 Geometry: New Functions

- line_2d_intersects
- line_2d_distance
- point_distance
- line_3d_distance
- ray_triangle_3d_intersects (for Qt3D Picking Detection)
