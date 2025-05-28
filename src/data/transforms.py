SPECIAL_PROCESSING_FUNCS = {}

def register_func(name):
    def decorator(fn):
        SPECIAL_PROCESSING_FUNCS[name] = fn
        return fn
    return decorator

@register_func("my_special_preprocessing")
def my_special_preprocessing(ds, cfg):
    # processing logic
    return ds
