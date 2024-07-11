

def convert_to_float(s):
    """
    Convert a string representation of a number in exponential format to a float.
    
    Parameters:
    - s: String containing the number in exponential format.
    
    Returns:
    - Float representation of the number.
    """
    try:
        # Check if the string contains a decimal point in the exponent part
        if 'E+' in s or 'E-' in s or 'e+' in s or 'e-' in s:
            parts = s.split('E') if 'E' in s else s.split('e')
            
            # Extract the base number and exponent
            base = float(parts[0])
            exponent = float(parts[1])
            
            # Adjust exponent if it contains a decimal point
            if '.' in parts[1]:
                exponent = int(float(parts[1]))  # Convert to int to remove decimal part
            
            # Calculate the final float value
            result = base * (10 ** exponent)
            
            return result
        
        # If no 'E' or 'e' found, convert directly to float
        return float(s)
    
    except ValueError as e:
        print(f"Error converting '{s}' to float: {e}")
        return None

