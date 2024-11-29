<!--Author:szuhsien.feng, Host:https://wiki.realtek.com/rest/api, SpaceKey:PKGSW, PageID:608178368, GitHub:https://github.com/Realtek-Package-Software-Group/RT-CKT-API/blob/main/CHANGE_LOG.md-->


# 2024.10

## 🎉 New Features
### `electrical_toolbox` module

This module is designed to provide some basic electrical calculation functions, currently including: inductance calculation for rectangular planes.

- `calculate_rectangular_cross_section_dc_self_inductance` ➜ Calculate the self-inductance of a rectangular plane.
- `calculate_rectangular_cross_section_dc_mutual_inductance` ➜ Calculate the mutual inductance between two rectangular planes.



# 2024.08

## 🎉 New Features
### `network` module

This module is designed to process touchstone files with basic property checking, including passivity/reciprocity/causality.
Also, it provides functions to calculate TDR profiles of the imported touchstone files.


- `NetworkData`: A class to deal with touchstone files.
    - `NetworkData.check_reciprocity` ➜ Check if the network is reciprocal.
    - `NetworkData.check_passivity` ➜ Check if the network is passive.
    - `NetworkData.check_causality` ➜ Check if the network is causal.
    - `NetworkData.calculate_tdr` ➜ Get TDR profile of the network.
