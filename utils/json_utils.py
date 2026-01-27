import json
import re
import logging

logger = logging.getLogger('json_utils')

def clean_json_string(text: str) -> str:
    """Clean common JSON issues (quotes, trailing commas)."""
    text = text.strip()
    text = text.replace("'", '"')  # single → double quotes
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # remove trailing commas
    return text

def robust_json_fix(text: str):
    """Try to fix JSON issues and parse safely."""
    try:
        fixed = clean_json_string(text)
        return json.loads(fixed)
    except Exception as e:
        logger.debug(f"robust_json_fix failed: {e}")
        return None

def extract_json_block(text: str):
    """Extract JSON inside markdown code fences."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# In json_utils.py, enhance the fix_json_issues function:
def fix_json_issues(json_string):
    """Attempt to fix common JSON parsing issues more robustly"""
    logger.debug("Attempting to fix JSON issues")
    
    try:
        # Remove any text before [ and after ]
        json_start = json_string.find('[')
        json_end = json_string.rfind(']') + 1
        if json_start != -1 and json_end != 0:
            json_string = json_string[json_start:json_end]
        
        # Remove markdown code block markers if present
        json_string = re.sub(r'^```(?:json)?\s*', '', json_string, flags=re.IGNORECASE)
        json_string = re.sub(r'\s*```$', '', json_string, flags=re.IGNORECASE)
        
        # Common JSON fixes
        fixes = [
            # NEW FIX: Add missing comma after an array if it's followed by a new key
            (r'\]\s*("correct_answer")', '], \\1'),
            # Fix missing commas between objects
            (r'\}\s*\{', '},{'),
            # Fix missing commas between array elements
            (r'"\s*"', '","'),
            # Fix unquoted keys
            (r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":'),
            # Fix single quoted values
            (r':\s*\'([^\']*?)\'\s*([,}\]])', r': "\1"\2'),
            # Fix trailing commas in arrays/objects
            (r',\s*]', ']'),
            (r',\s*}', '}'),
            # Fix missing quotes around option values
            # NOTE: Requires the fix_options_array helper function
            (r'"options"\s*:\s*\[\s*([^\[\]]*?)\s*\]', lambda m: f'"options": [{fix_options_array(m.group(1))}]'),
        ]
        
        for pattern, replacement in fixes:
            if callable(replacement):
                json_string = re.sub(pattern, replacement, json_string)
            else:
                json_string = re.sub(pattern, replacement, json_string)
            
        return json_string
        
    except Exception as e:
        logger.error(f"Error fixing JSON: {e}")
        return json_string
    

def fix_options_array(options_text):
    """Fix issues in options arrays specifically"""
    # Split options by comma but be careful about commas inside quotes
    options = []
    current_option = ""
    in_quotes = False
    escape_next = False
    
    for char in options_text:
        if escape_next:
            current_option += char
            escape_next = False
        elif char == '\\':
            current_option += char
            escape_next = True
        elif char == '"':
            current_option += char
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            options.append(current_option.strip())
            current_option = ""
        else:
            current_option += char
    
    if current_option:
        options.append(current_option.strip())
    
    # Ensure each option is properly quoted
    fixed_options = []
    for option in options:
        option = option.strip()
        if not option:
            continue
        if not (option.startswith('"') and option.endswith('"')):
            # Add quotes if missing
            option = f'"{option}"'
        fixed_options.append(option)
    
    return ', '.join(fixed_options)
