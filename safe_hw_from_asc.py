import os
import re

def safe_hw_from_asc(fn_asc, verbose=False):
    """
    Reads hardware specifications from an .asc file, handling dependencies.
    
    Args:
        fn_asc (str): Path to the .asc file.
        verbose (bool): Whether to print verification steps.
        
    Returns:
        dict: A dictionary containing the hardware specifications.
    """
    
    # 1. Initialize Structure
    # fileparts in MATLAB returns path, name, ext
    dir_name, file_name = os.path.split(fn_asc)
    base_name, ext = os.path.splitext(file_name)
    
    hw = {
        'name': base_name,
        'fn_asc': fn_asc,
        'x': {}, 'y': {}, 'z': {}, # Initialize sub-dicts to avoid KeyErrors
        'dependency': None,
        'checksum': None
    }

    # 2. Read and Preprocess File
    try:
        with open(fn_asc, 'r', encoding='utf-8', errors='ignore') as f:
            raw_str = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {fn_asc}")
        return hw

    # MATLAB: STR = STR(~isspace(STR)); (Removes ALL whitespace: spaces, tabs, newlines)
    clean_str = "".join(raw_str.split()) 
    lower_str = clean_str.lower()

    # 3. Define Mapping List
    # Format: (Search String, Category/Key, SubKey)
    str_list = [
        ('checksum',                         'checksum', ''),
        ('$INCLUDE',                         'dependency', ''),
        # Support both with and without namespace prefix
        ('flGSWDStimulationLimitX=',         'x', 'stim_limit'),
        ('flGSWDStimulationLimitY=',         'y', 'stim_limit'),
        ('flGSWDStimulationLimitZ=',         'z', 'stim_limit'),

        ('flGSWDStimulationThresholdX=',     'x', 'stim_thresh'),
        ('flGSWDStimulationThresholdY=',     'y', 'stim_thresh'),
        ('flGSWDStimulationThresholdZ=',     'z', 'stim_thresh'),

        ('flGSWDTauX[0]=',                   'x', 'tau1'),
        ('flGSWDTauX[1]=',                   'x', 'tau2'),
        ('flGSWDTauX[2]=',                   'x', 'tau3'),
        ('flGSWDTauY[0]=',                   'y', 'tau1'),
        ('flGSWDTauY[1]=',                   'y', 'tau2'),
        ('flGSWDTauY[2]=',                   'y', 'tau3'),
        ('flGSWDTauZ[0]=',                   'z', 'tau1'),
        ('flGSWDTauZ[1]=',                   'z', 'tau2'),
        ('flGSWDTauZ[2]=',                   'z', 'tau3'),

        ('flGSWDAX[0]=',                     'x', 'a1'),
        ('flGSWDAX[1]=',                     'x', 'a2'),
        ('flGSWDAX[2]=',                     'x', 'a3'),
        ('flGSWDAY[0]=',                     'y', 'a1'),
        ('flGSWDAY[1]=',                     'y', 'a2'),
        ('flGSWDAY[2]=',                     'y', 'a3'),
        ('flGSWDAZ[0]=',                     'z', 'a1'),
        ('flGSWDAZ[1]=',                     'z', 'a2'),
        ('flGSWDAZ[2]=',                     'z', 'a3'),

        ('GScaleFactorX=',                   'x', 'g_scale'),
        ('GScaleFactorY=',                   'y', 'g_scale'),
        ('GScaleFactorZ=',                   'z', 'g_scale'),
    ]

    # 4. Parsing Loop
    for search_term, category, key in str_list:
        # MATLAB: strfind(str, lower(...))
        search_term_lower = search_term.lower()
        idx = lower_str.find(search_term_lower)
        
        if idx != -1:
            # Move index to the end of the matched string
            start_idx = idx + len(search_term_lower)
            
            # Extract content based on type
            if search_term == 'checksum':
                # Match digits
                match = re.search(r'\d+', lower_str[start_idx:])
                if match:
                    hw[category] = match.group(0)
                    
            elif search_term == '$INCLUDE':
                # Match word characters (filename). Note: MATLAB uses original case string here.
                # However, since we stripped whitespace, indices align between clean_str and lower_str.
                match = re.search(r'\w+', clean_str[start_idx:])
                if match:
                    hw[category] = match.group(0)
                    
            else:
                # Match float numbers (equivalent to %g)
                # Matches: 123, -123.45, 1.23e-5
                match = re.search(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', lower_str[start_idx:])
                if match:
                    try:
                        val = float(match.group(0))
                        if key:
                            hw[category][key] = val
                        else:
                            hw[category] = val
                    except ValueError:
                        pass

    # 5. Handle Dependencies
    if hw.get('dependency'):
        if verbose:
            print(f"Reading PNS parameters from other system ({hw['name']} is based on {hw['dependency']})")
        
        # Construct path: same dir, new name, same extension
        # MATLAB: fn_asc_dep = [a filesep hw.dependency c];
        fn_asc_dep = os.path.join(dir_name, hw['dependency'] + ext)
        
        # Recursive call
        hw_dep = safe_hw_from_asc(fn_asc_dep, verbose)
        
        # Merge: hw overrides hw_dep (or fills in gaps). 
        # The MATLAB function name 'mergeIfNotEmpty' implies we fill hw with hw_dep 
        # where hw is missing data.
        hw = _safe_hw_merge_if_not_empty(hw, hw_dep)

    # 6. Verification (Placeholder logic)
    if verbose:
        try:
            # These functions were not provided, so we simulate the call
            _safe_hw_verify(hw)
            _safe_hw_check(hw)
        except Exception as e:
            print(f"Verification error: {e}")

    return hw

# --- Helper Functions ---

def _safe_hw_merge_if_not_empty(primary, secondary):
    """
    Merges secondary into primary recursively. 
    Keys in primary are NOT overwritten by secondary.
    Equivalent to filling in gaps in 'primary' using 'secondary'.
    """
    for k, v in secondary.items():
        if k not in primary or primary[k] is None:
            primary[k] = v
        elif isinstance(v, dict) and isinstance(primary.get(k), dict):
            _safe_hw_merge_if_not_empty(primary[k], v)
    return primary

def _safe_hw_verify(hw):
    # Placeholder for validation logic
    pass

def _safe_hw_check(hw):
    # Placeholder for check logic
    pass

# --- Usage Example ---
if __name__ == "__main__":
    # Example usage (assuming a file exists)
    # result = safe_hw_from_asc('my_hardware_file.asc', verbose=True)
    # print(result)
    pass