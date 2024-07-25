

def replace_char_with_count(input_string, char_to_search):
    # Escape the character to search to avoid issues with special regex characters
    pattern = f"({re.escape(char_to_search)}{{2,}})"
    
    # Find all groups of consecutive occurrences of the character
    matches = re.findall(pattern, input_string)
    
    # For each match, replace it with the count and the character
    for match in matches:
        count = len(match)
        replacement = f"{count}{char_to_search[0]}"
        input_string = input_string.replace(match, replacement, 1)
    
    return input_string

import re

