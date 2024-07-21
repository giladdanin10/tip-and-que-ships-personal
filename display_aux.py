
# print dictionary vertically
def print_dict(dict):
    for key in dict.keys():
        print(f'{key}:{dict[key]}')

def print_dict_lines(param_dict):
    for key, value in param_dict.items():
        print(f"'{key}': {value},")
