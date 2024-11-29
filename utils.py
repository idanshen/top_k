def is_correct_answer(official_answer: str, generated_answer: str) -> bool:
    """
    Compare the official answer and the generated answer for the GSM8K dataset.
    
    This function extracts numerical results from both answers and determines
    if they match within a specified tolerance.

    Returns True if the numerical answers are sufficiently close, False otherwise.
    """
    import re

    def extract_number(text: str) -> float:
        """
        Extract the last numerical value from the given text.
        
        Args:
            text (str): The text to extract the number from.
        
        Returns:
            float: The extracted number. Returns None if no number is found.
        """
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        return float(matches[-1]) if matches else None

    official_num = extract_number(official_answer)
    generated_num = extract_number(generated_answer)

    if official_num is None or generated_num is None:
        return False

    # Define a tolerance for numerical comparison
    tolerance = 1e-3
    return abs(official_num - generated_num) < tolerance

def extract_final_answer(text: str) -> str:
    """
    Extract the final answer from a string that contains arithmetic expressions and a final answer.
    
    Args:
        text (str): The text containing arithmetic expressions and final answer marked with ####
    
    Returns:
        str: The extracted final answer, or empty string if no answer is found
    """
    import re
    
    # Look for pattern #### followed by number/text at the end of string
    match = re.search(r'####\s*(\S+)\s*$', text)
    if match:
        return match.group(1)
    return ""
