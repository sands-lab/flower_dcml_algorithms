def construct_config_fn(config_dict):
    def f(_):
        return config_dict
    return f
