import traceback
import inspect

def parse_func_params(params, default_params):
    parsed_params = {}

    # Get the name of the calling function
    calling_func_name = inspect.currentframe().f_back.f_code.co_name
    
    # Validate and parse each parameter
    for param_name, param_info in default_params.items():
        if isinstance(param_info, dict):
            default_value = param_info.get('default')
            allowed_values = param_info.get('optional', [])
        else:
            default_value = param_info
            allowed_values = []

        if param_name in params:
            param_value = params[param_name]
        else:
            param_value = default_value

        # Validate parameter value against allowed_values if provided
        if allowed_values and param_value not in allowed_values:
            raise ValueError(f"{calling_func_name}: Invalid value '{param_value}' for parameter '{param_name}'. Allowed values are {sorted(allowed_values)}.")

        parsed_params[param_name] = param_value

    return parsed_params



def func1(**params):
    # Define default and optional values for each parameter in default_params
    default_params = {
        'a': {'default': 5, 'optional': {5, 7, 3}},
        'b': {'default': 10, 'optional': {2, 5, 10}}
    }

    try:
        params = parse_func_params(params, default_params)
    except ValueError as e:
        print(e)  # Print the exception message with calling stack path
        return None

    # Perform calculations using parsed parameters
    a = params['a']
    b = params['b']
    c = a + b
    return c

def func2(e,g,**params):
    # Define default and optional values for each parameter in default_params
    default_params = {
        'a': 5,
        'd': {'default': 2, 'optional': {2, 5, 10}}
    }

    try:
        params = parse_func_params(params, default_params)
    except ValueError as e:
        print(e)  # Print the exception message with calling stack path
        return None
    
    # Call func1 with validated params and perform additional calculation
    f = func1(**params) + params['d']+e

    return f