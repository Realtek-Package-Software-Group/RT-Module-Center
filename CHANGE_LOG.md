<!--Author:jeff.chou, Host:https://wiki.realtek.com/rest/api, SpaceKey:PKGSW, PageID:539913360, GitHub:https://github.com/Realtek-Package-Software-Group/RT-Math-API/blob/main/CHANGE_LOG.md-->

# 2024.08


## 🎉 New Features

<h3><code>network</code>  module</h3>

This module is designed to process touchstone files with basic property checking, including passivity/reciprocity/causality.
Also, it provides functions to calculate TDR profiles of the imported touchstone files.


- `NetworkData`: A class to deal with touchstone files.
    - `NetworkData.check_reciprocity` ➜ Check if the network is reciprocal.
    - `NetworkData.check_passivity` ➜ Check if the network is passive.
    - `NetworkData.check_causality` ➜ Check if the network is causal.
    - `NetworkData.calculate_tdr` ➜ Get TDR profile of the network.
