def get_anon_level_from_config(module_config, dataset_name):
    for k, v in module_config.items():
        if k.startswith("anon_level_"):
            for dname_short in v:
                if dname_short in dataset_name:
                    return k.replace("anon_level_", "")
    raise ValueError("anon_level not implemented")
