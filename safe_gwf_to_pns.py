import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
import safe_hw_from_asc

def safe_example_hw():
    """
    SAFE model parameters for EXAMPLE scanner hardware (not a real scanner).
    
    Returns
    -------
    hw : dict
        Dictionary containing hardware parameters for x, y, and z axes.
    """
    hw = {
        'name': 'MP_GPA_EXAMPLE',
        'checksum': '1234567890',
        'dependency': '',
        'x': {
            'tau1': 0.20,       # ms
            'tau2': 0.03,       # ms
            'tau3': 3.00,       # ms
            'a1': 0.40,
            'a2': 0.10,
            'a3': 0.50,
            'stim_limit': 30.0, # T/m/s
            'stim_thresh': 24.0,# T/m/s
            'g_scale': 0.35     # 1
        },
        'y': {
            'tau1': 1.50,       # ms
            'tau2': 2.50,       # ms
            'tau3': 0.15,       # ms
            'a1': 0.55,
            'a2': 0.15,
            'a3': 0.30,
            'stim_limit': 15.0, # T/m/s
            'stim_thresh': 12.0,# T/m/s
            'g_scale': 0.31     # 1
        },
        'z': {
            'tau1': 2.00,       # ms
            'tau2': 0.12,       # ms
            'tau3': 1.00,       # ms
            'a1': 0.42,
            'a2': 0.40,
            'a3': 0.18,
            'stim_limit': 25.0, # T/m/s
            'stim_thresh': 20.0,# T/m/s
            'g_scale': 0.25     # 1
        }
    }
    return hw

def safe_longest_time_const(hw):
    """Get the longest time constant to estimate zero padding size."""
    taus = []
    for axis in ['x', 'y', 'z']:
        if axis in hw:
            taus.extend([hw[axis]['tau1'], hw[axis]['tau2'], hw[axis]['tau3']])
    return max(taus)

def safe_hw_check(hw):
    """Make sure that all is well with the hardware configuration."""
    # Check if weights sum to 1
    for axis in ['x', 'y', 'z']:
        total_a = hw[axis]['a1'] + hw[axis]['a2'] + hw[axis]['a3']
        if abs(total_a - 1.0) > 0.001:
            raise ValueError(f'Hardware specification {axis}: a1+a2+a3 must be equal to 1!')

    required_params = ['stim_limit', 'stim_thresh', 'tau1', 'tau2', 'tau3', 'a1', 'a2', 'a3', 'g_scale']
    
    for axis in ['x', 'y', 'z']:
        for param in required_params:
            if param not in hw[axis] or hw[axis][param] is None:
                raise ValueError(f"Hardware specification {axis}.{param} is empty or missing!")

def safe_tau_lowpass(dgdt, tau, dt):
    """
    Apply a RC lowpass filter with time constant tau = RC.
    NOTE: tau and dt need to be in the same unit (i.e. s or ms)
    """
    # Filter coefficient alpha = dt / (tau + dt)
    alpha = dt / (tau + dt)
    
    # Python's lfilter implements: a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] ... - a[1]*y[n-1] ...
    # The MATLAB loop was: y[n] = alpha*x[n] + (1-alpha)*y[n-1]
    # This corresponds to: y[n] - (1-alpha)*y[n-1] = alpha*x[n]
    # So a = [1, -(1-alpha)], b = [alpha]
    
    b = [alpha]
    a = [1, -(1 - alpha)]
    
    # Apply filter along the first axis (time)
    return lfilter(b, a, dgdt, axis=0)

def safe_pns_model(dgdt, dt, hw_axis):
    """
    Calculate PNS stimulation for a specific axis.
    dgdt is in T/m/s, dt is in s.
    """
    # Convert dt to ms for the lowpass filter since tau is in ms
    dt_ms = dt * 1000.0
    
    # Term 1: a1 * abs(LowPass(dgdt))
    lp1 = safe_tau_lowpass(dgdt, hw_axis['tau1'], dt_ms)
    stim1 = hw_axis['a1'] * np.abs(lp1)
    
    # Term 2: a2 * LowPass(abs(dgdt))
    lp2 = safe_tau_lowpass(np.abs(dgdt), hw_axis['tau2'], dt_ms)
    stim2 = hw_axis['a2'] * lp2
    
    # Term 3: a3 * abs(LowPass(dgdt))
    lp3 = safe_tau_lowpass(dgdt, hw_axis['tau3'], dt_ms)
    stim3 = hw_axis['a3'] * np.abs(lp3)
    
    # Combine
    stim = (stim1 + stim2 + stim3) / hw_axis['stim_thresh'] * hw_axis['g_scale'] * 100.0
    return stim

def safe_gwf_to_pns(gwf, rf, dt, hw, do_padding=True):
    """
    Main function to convert Gradient Waveform to PNS.
    
    gwf: (nx3) array in T/m
    rf:  (nx1) array (or similar)
    dt:  float, sampling time in s
    hw:  hardware dictionary
    """
    gwf = np.array(gwf)
    rf = np.array(rf)
    
    # 1. Zero Padding
    if do_padding:
        zpt = safe_longest_time_const(hw) * 4 / 1000.0 # convert ms to s
        pad_len_pre = int(np.round(zpt / 4 / dt))
        pad_len_post = int(np.round(zpt / 1 / dt))
        
        # Create zero arrays
        zp1 = np.zeros((pad_len_pre, 3))
        zp2 = np.zeros((pad_len_post, 3))
        
        # Pad GWF
        gwf = np.concatenate([zp1, gwf, zp2], axis=0)
        
        # Pad RF (Handle 1D or 2D RF arrays)
        if rf.ndim == 1:
            rf_zp1 = np.zeros(pad_len_pre)
            rf_zp2 = np.zeros(pad_len_post)
            rf = np.concatenate([rf_zp1, rf, rf_zp2])
        else:
            rf_zp1 = np.zeros((pad_len_pre, rf.shape[1]))
            rf_zp2 = np.zeros((pad_len_post, rf.shape[1]))
            rf = np.concatenate([rf_zp1, rf, rf_zp2], axis=0)

    # 2. Hardware Check
    safe_hw_check(hw)
    
    # 3. Calculate Slew Rate (dgdt)
    # MATLAB diff(gwf, 1) calculates difference along dim 1.
    # Note: This reduces the array length by 1.
    dgdt = np.diff(gwf, axis=0) / dt
    
    # 4. Calculate PNS for each axis
    pns = np.zeros_like(dgdt)
    axes_keys = ['x', 'y', 'z']
    
    for i, ax_name in enumerate(axes_keys):
        # Pass the specific column for the axis
        pns[:, i] = safe_pns_model(dgdt[:, i], dt, hw[ax_name])
        
    # 5. Pack results
    res = {
        'pns': pns,
        'gwf': gwf,
        'rf': rf,
        'dgdt': dgdt,
        'dt': dt,
        'hw': hw
    }
    
    return pns, res

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Load Hardware
    hw = safe_hw_from_asc.safe_hw_from_asc('safe_pns_prediction/MP_GradSys_K2298_2250V_1250A_W60_SC72CD.asc')
    # or use example hardware: hw = safe_example_hw()
    
    # 2. Load Data provided by user
    dt = 1e-3  # seconds (1 ms)

    # Raw normalized data matrix
    raw_data = np.array([
        [ 0,         0,         0],
        [-0.2005,    0.9334,    0.3029],
        [-0.2050,    0.9324,    0.3031],
        [-0.2146,    0.9302,    0.3032],
        [-0.2313,    0.9263,    0.3030],
        [-0.2589,    0.9193,    0.3019],
        [-0.3059,    0.9060,    0.2980],
        [-0.3892,    0.8767,    0.2883],
        [-0.3850,    0.7147,    0.3234],
        [-0.3687,    0.5255,    0.3653],
        [-0.3509,    0.3241,    0.4070],
        [-0.3323,    0.1166,    0.4457],
        [-0.3136,   -0.0906,    0.4783],
        [-0.2956,   -0.2913,    0.5019],
        [-0.2790,   -0.4793,    0.5139],
        [-0.2642,   -0.6491,    0.5118],
        [-0.2518,   -0.7957,    0.4939],
        [-0.2350,   -0.8722,    0.4329],
        [-0.2187,   -0.9111,    0.3541],
        [-0.2063,   -0.9409,    0.2747],
        [-0.1977,   -0.9627,    0.1933],
        [-0.1938,   -0.9768,    0.1080],
        [-0.1967,   -0.9820,    0.0159],
        [-0.2114,   -0.9751,   -0.0883],
        [-0.2292,   -0.9219,   -0.2150],
        [-0.2299,   -0.8091,   -0.3561],
        [-0.2290,   -0.6748,   -0.5011],
        [-0.2253,   -0.5239,   -0.6460],
        [-0.2178,   -0.3620,   -0.7868],
        [-0.2056,   -0.1948,   -0.9194],
        [-0.1391,   -0.0473,   -0.9908],
        [-0.0476,    0.0607,   -0.9987],
        [ 0.0215,    0.1452,   -0.9909],
        [ 0.0725,    0.2136,   -0.9759],
        [ 0.1114,    0.2709,   -0.9579],
        [ 0.1426,    0.3204,   -0.9383],
        [ 0.1690,    0.3641,   -0.9177],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [ 0,         0,         0],
        [-0.3734,   -0.1768,    0.9125],
        [-0.3825,   -0.2310,    0.8965],
        [-0.3919,   -0.2895,    0.8752],
        [-0.4015,   -0.3543,    0.8465],
        [-0.4108,   -0.4290,    0.8065],
        [-0.4182,   -0.5202,    0.7469],
        [-0.4178,   -0.6423,    0.6451],
        [-0.3855,   -0.8173,    0.4321],
        [-0.3110,   -0.9418,    0.1401],
        [-0.2526,   -0.9669,   -0.0674],
        [-0.2100,   -0.9541,   -0.2213],
        [-0.1766,   -0.9227,   -0.3474],
        [-0.1491,   -0.8788,   -0.4570],
        [-0.1258,   -0.8239,   -0.5555],
        [-0.1056,   -0.7583,   -0.6459],
        [-0.0882,   -0.6809,   -0.7293],
        [-0.0734,   -0.5900,   -0.8061],
        [-0.0615,   -0.4830,   -0.8753],
        [-0.0533,   -0.3556,   -0.9349],
        [-0.0506,   -0.2005,   -0.9801],
        [-0.0575,   -0.0019,   -1.0000],
        [-0.0909,    0.2976,   -0.9521],
        [-0.3027,    0.9509,   -0.0860],
        [-0.2737,    0.9610,   -0.0692],
        [-0.2524,    0.9675,   -0.0596],
        [-0.2364,    0.9719,   -0.0533],
        [-0.2245,    0.9749,   -0.0490],
        [-0.2158,    0.9770,   -0.0459],
        [-0.2097,    0.9785,   -0.0439],
        [-0.2058,    0.9794,   -0.0426],
        [-0.2039,    0.9798,   -0.0420],
        [ 0,         0,         0]
    ])

    # Apply scaling factor
    gwf_input = 0.08 * raw_data

    # Create RF
    # MATLAB: rf = ones(length(gwf),1); rf(41:end) = -1;
    # Python indices start at 0. MATLAB index 41 is Python index 40.
    rf_input = np.ones(len(gwf_input))
    rf_input[40:] = -1

    # 3. Run Calculation
    try:
        pns, res = safe_gwf_to_pns(gwf_input, rf_input, dt, hw, do_padding=True)
        print("PNS Calculated successfully.")
        
        # --- Plotting ---
        gwf_plot = res['gwf']
        pns_plot = res['pns']
        rf_plot  = res['rf']
        
        # Create time vectors
        # Note: gwf has padding added inside the function, so we must use res['gwf'] length
        t_gwf = np.arange(len(gwf_plot)) * dt * 1000 # ms
        t_pns = np.arange(len(pns_plot)) * dt * 1000 # ms

        # Align lengths (PNS is shorter by 1 due to diff)
        gwf_plot = gwf_plot[:-1, :]
        rf_plot = rf_plot[:-1]
        t_gwf = t_gwf[:-1]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        
        # Plot 1: Gradients
        ax1.plot(t_gwf, gwf_plot[:, 0] * 1000, label='Gx', color='r')
        ax1.plot(t_gwf, gwf_plot[:, 1] * 1000, label='Gy', color='g')
        ax1.plot(t_gwf, gwf_plot[:, 2] * 1000, label='Gz', color='b')
        ax1.set_ylabel('Gradient [mT/m]')
        ax1.set_title('Gradient Waveforms')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')
        
        # Optional: Plot RF on twin axis to see the flip
        ax1_rf = ax1.twinx()
        ax1_rf.plot(t_gwf, rf_plot, color='k', linestyle=':', alpha=0.3, label='RF')
        ax1_rf.set_ylabel('RF State')
        
        # Plot 2: PNS
        ax2.plot(t_pns, pns_plot[:, 0], label='PNS x', color='r', linestyle='--')
        ax2.plot(t_pns, pns_plot[:, 1], label='PNS y', color='g', linestyle='--')
        ax2.plot(t_pns, pns_plot[:, 2], label='PNS z', color='b', linestyle='--')
        
        # Norm
        pns_norm = np.linalg.norm(pns_plot, axis=1)
        #ax2.plot(t_pns, pns_norm, label='|PNS|', color='k', linewidth=1.5)
        
        ax2.set_ylabel('Stimulation [%]')
        ax2.set_xlabel('Time [ms]')
        ax2.set_title('Peripheral Nerve Stimulation (PNS)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")