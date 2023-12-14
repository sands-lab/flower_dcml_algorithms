def construct_config_fn(config_dict):
    def f(epoch):
        return config_dict
    return f