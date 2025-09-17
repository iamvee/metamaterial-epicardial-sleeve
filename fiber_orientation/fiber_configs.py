data = {
    "job_name": "heat_transfer_for_fiber_orientation",
    "nodes_file": "AllNodes.txt",
    "elements_file": "AllElements.txt",
    "element_type": "DC3D4",
    "set_files": ["LVSets.txt", "RVSets.txt", "EPISets.txt", "BVSets.txt"],
    "solid_section_set": "Set-BVCell",
    "material_name": "my_iso_heat",
    "conductivity": "1.0",
    "step_name": "initialization",
    "nlgeom": "NO",
    "inc": 1000000,
    "deltmx": 0.0,
    "tolerance": 1.0e-5,
    "restart_frequency": 0,
    "history_frequency": 0,
    "time_interval": 1.0,
    "node_outputs": ["CFL", "NT", "RFL", "RFLE"],
    "element_outputs": ["FLUXS", "HBF", "HFL", "NFLUX", "TEMP"],
    "el_print_outputs": ["TEMP", "HFL"]
}

data_b = {
    "boundaries": [
        {"set_name": "Set-LV_ENDO", "direction": 11, "value": 0.0},
        {"set_name": "Set-RV_ENDO", "direction": 11, "value": 0.0},
        {"set_name": "Set-EPI", "direction": 11, "value": 10.0}
    ],
    **data
}

data_c = {
    "boundaries": [
        {"set_name": "Set-LV_ENDO", "direction": 11, "value": 10.0},
        {"set_name": "Set-RV_ENDO", "direction": 11, "value": 0.0},
        {"set_name": "Set-EPI", "direction": 11, "value": 0.0}
    ],
    **data
}

data_d = {
    "boundaries": [
        {"set_name": "Set-RV_ENDO", "direction": 11, "value": 10.0},
        {"set_name": "Set-LV_ENDO", "direction": 11, "value": 0.0},
        {"set_name": "Set-EPI", "direction": 11, "value": 0.0}
    ],
    **data
}
