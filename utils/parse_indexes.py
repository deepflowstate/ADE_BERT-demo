import re
import ast

def parse_indexes(indexes_str):
    """
    Converts a string representation of a NumPy array with optional dtype info
    into a native Python list by cleaning the string and evaluating it.
    
    Args:
        indexes_str (str): String representing a NumPy array (e.g. "array([1, 2], dtype=int32)").
        
    Returns:
        list: Corresponding Python list of indexes.
    """
    cleaned = re.sub(r"array\((\[.*?\])(?:, dtype=\w+)?\)", r"\1", indexes_str)
    cleaned = re.sub(r",?\s*dtype=\w+", "", cleaned)
    cleaned = cleaned.replace(")", "")
    return ast.literal_eval(cleaned)