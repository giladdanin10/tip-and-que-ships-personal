import traceback

def parse_parameter(parameter, allowed_values):
    """
    Example function that only accepts specific values for its parameter.

    Parameters:
    - parameter: str, the input parameter which must be one of the allowed values.
    - allowed_values: list, the set of allowed values for the parameter.

    Raises:
    - ValueError: if the parameter is not in the allowed values.
    """
    
    
    if (isinstance(parameter,list)):
        for value in parameter:
            if value not in allowed_values:
                raise ValueError(f"Invalid value '{value}'. Allowed values are: {allowed_values}")
    else: 
        if parameter not in allowed_values:
            raise ValueError(f"Invalid value '{parameter}'. Allowed values are: {allowed_values}")
    
    

def get_sfunc_path():
    stack = traceback.format_stack()



