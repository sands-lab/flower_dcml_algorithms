def construct_config_fn(config_dict, evaluator):
    def f(_):
        if evaluator is not None:
            evaluator.epoch += 1
        return config_dict
    return f
