# Usage



## python post-proc

```python
fig, ax = plt.subplots(figsize=(10, 6))


input_params_dict_extend = [
    {
        'inp_file': "./MI_lv_416_inf_649/model.inp",
        'dat_file': "./MI_lv_416_inf_649/MI_lv_416_inf_649.dat",
        'reference_point': (0, 0, 0.0),
        'surface_name': 'LVENDO',
        'nodes_used_set': ['ENDO_NODES'],
        'nset_name_for_endo': ['ENDO_NODES'],

    },
    { 
        'inp_file': "./A2_lv_416_inf_416_zeta_5/model.inp",
        'dat_file': "./A2_lv_416_inf_416_zeta_5/A2_lv_416_inf_416_zeta_5.dat",
        'reference_point': (0, 0, 0.0),
        'surface_name': 'LVENDO',
        'nodes_used_set': ['NS_LVENDO'],
        'nset_name_for_endo': ['NS_LVENDO'],
    }
]

for params in input_params_dict_extend:
    params['analyzer'] = AbaqusVolumeAnalyzer(
        inp_file=params['inp_file'],
        dat_file=params['dat_file']
    )

    params['volume_history_of_steps'] = {
        f'step_{step_num}': params['analyzer'].analyze_volume_history(
            surface_name=params['surface_name'],
            step_num=step_num,
            reference_point=params['reference_point']
        ) for step_num in [1, 2]
    }

    _df = params['volume_history_of_steps']['step_2']
    _t = _df['fraction_completed']
    _v = _df['displaced_volume']
    _p_f = params['analyzer'].abaqus_input.interp_data.get('amplitude', None)
    _p = _p_f(_t)

    ax.plot(_v, _p,  label=f"{params['inp_file']} - {params['surface_name']}",  marker='o')

```