import torch
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
        'x': {'tau1': 0.20, 'tau2': 0.03, 'tau3': 3.00, 'a1': 0.40, 'a2': 0.10, 'a3': 0.50,
              'stim_limit': 30.0, 'stim_thresh': 24.0, 'g_scale': 0.35},
        'y': {'tau1': 1.50, 'tau2': 2.50, 'tau3': 0.15, 'a1': 0.55, 'a2': 0.15, 'a3': 0.30,
              'stim_limit': 15.0, 'stim_thresh': 12.0, 'g_scale': 0.31},
        'z': {'tau1': 2.00, 'tau2': 0.12, 'tau3': 1.00, 'a1': 0.42, 'a2': 0.40, 'a3': 0.18,
              'stim_limit': 25.0, 'stim_thresh': 20.0, 'g_scale': 0.25}
    }
    return hw


def safe_longest_time_const(hw):
    taus = []
    for axis in ['x', 'y', 'z']:
        if axis in hw:
            taus.extend([hw[axis]['tau1'], hw[axis]['tau2'], hw[axis]['tau3']])
    return max(taus)

def safe_hw_check(hw):
    for axis in ['x', 'y', 'z']:
        total_a = hw[axis]['a1'] + hw[axis]['a2'] + hw[axis]['a3']
        if abs(total_a - 1.0) > 0.001:
            raise ValueError(f'Hardware specification {axis}: a1+a2+a3 must be equal to 1!')
    required_params = ['stim_limit', 'stim_thresh', 'tau1', 'tau2', 'tau3', 'a1', 'a2', 'a3', 'g_scale']
    for axis in ['x', 'y', 'z']:
        for param in required_params:
            if param not in hw[axis] or hw[axis][param] is None:
                raise ValueError(f"Hardware specification {axis}.{param} is empty or missing!")


def safe_tau_lowpass_torch(dgdt: torch.Tensor, tau: float, dt: float, cutoff: float = 0.005) -> torch.Tensor:
    """
    Differentiable RC lowpass filter (like MATLAB loop).
    dgdt: (time,)
    tau: time constant in ms
    dt: sampling interval in ms
    cutoff: stopping criterion
    """

    print("Applying lowpass filter...", flush=True)
    alpha = dt / (tau + dt)
    fw = torch.zeros_like(dgdt)
    fw[0] = alpha * dgdt[0]

    shift = params['timesteps'] - 1 #understand if -1 or not

    #make the estimation of max t beforehand
    for t in range(1, dgdt.shape[0]): 
        fw[t] = alpha * dgdt[t] + (1 - alpha) * fw[t - 1]

        # phi = C * exp(-t*dt/tau)
        if t >= shift:
            C = torch.abs(fw[shift] - fw[0])
            exponent = torch.tensor(-t * dt / tau, device=dgdt.device, dtype=dgdt.dtype)
            phi = C * torch.exp(exponent)

            if phi <= cutoff:
                print(f"Stopping at t = {t} ms, phi = {phi.item():.6f}")
                return fw[:t+1]
    return fw


# not sure about this. now not used
def safe_tau_lowpass_torch_vec(dgdt: torch.Tensor, tau: float, dt: float) -> torch.Tensor:
    """
    Vectorized differentiable RC lowpass filter using PyTorch.
    dgdt: (time,)
    tau: time constant in ms
    dt: sampling interval in ms
    """
    alpha = dt / (tau + dt)
    time_len = dgdt.shape[0]

    # Compute filter weights (exponentially decaying)
    # Recursive formula: y[n] = alpha * x[n] + (1-alpha) * y[n-1]
    # Closed form: y[n] = sum_{k=0}^{n} alpha*(1-alpha)^k * x[n-k]
    # Equivalent: convolve x with kernel (alpha*(1-alpha)^k)
    
    # Create the kernel (length = time_len)
    kernel = alpha * (1 - alpha) ** torch.arange(time_len, device=dgdt.device, dtype=dgdt.dtype)
    
    # Convolve with dgdt using cumulative sum trick for efficiency
    # Flip kernel for causal convolution
    y = torch.nn.functional.conv1d(
        dgdt.view(1,1,-1),
        kernel.view(1,1,-1),
        padding=0
    ).view(-1)
    
    return y


def safe_pns_model_torch(dgdt: torch.Tensor, dt: float, hw_axis: dict) -> torch.Tensor:
    dt_ms = dt * 1000.0  # convert to ms
    lp1 = safe_tau_lowpass_torch(dgdt, hw_axis['tau1'], dt_ms)
    stim1 = hw_axis['a1'] * torch.abs(lp1)
    lp2 = safe_tau_lowpass_torch(torch.abs(dgdt), hw_axis['tau2'], dt_ms)
    stim2 = hw_axis['a2'] * lp2
    lp3 = safe_tau_lowpass_torch(dgdt, hw_axis['tau3'], dt_ms)
    stim3 = hw_axis['a3'] * torch.abs(lp3)
    stim = (stim1 + stim2 + stim3) / hw_axis['stim_thresh'] * hw_axis['g_scale'] * 100.0
    return stim


def safe_gwf_to_pns_torch(gwf: torch.Tensor, rf: torch.Tensor, dt: float, hw: dict, do_padding=True):
    gwf = gwf.to(torch.float32)
    rf = rf.to(torch.float32)

    # Zero Padding
    if do_padding:
        zpt = safe_longest_time_const(hw) * 4 / 1000.0
        pad_len_pre = int(round(zpt / 4 / dt))
        pad_len_post = int(round(zpt / 1 / dt))
        gwf = torch.cat([torch.zeros((pad_len_pre, 3), device=gwf.device),
                         gwf,
                         torch.zeros((pad_len_post, 3), device=gwf.device)], dim=0)
        if rf.ndim == 1:
            rf = torch.cat([torch.zeros(pad_len_pre, device=rf.device),
                            rf,
                            torch.zeros(pad_len_post, device=rf.device)], dim=0)
        else:
            rf = torch.cat([torch.zeros((pad_len_pre, rf.shape[1]), device=rf.device),
                            rf,
                            torch.zeros((pad_len_post, rf.shape[1]), device=rf.device)], dim=0)

    # Hardware check
    safe_hw_check(hw)

    # Slew rate
    dgdt = (gwf[1:] - gwf[:-1]) / dt

    # PNS calculation
    pns = torch.zeros_like(dgdt)
    for i, ax in enumerate(['x','y','z']):
        pns[:, i] = safe_pns_model_torch(dgdt[:, i], dt, hw[ax])

    res = {
        'pns': pns,
        'gwf': gwf,
        'rf': rf,
        'dgdt': dgdt,
        'dt': dt,
        'hw': hw
    }
    return pns, res


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Hardware
    try:
        hw = safe_hw_from_asc.safe_hw_from_asc('safe_pns_prediction/MP_GradSys_K2298_2250V_1250A_W60_SC72CD.asc')
    except:
        hw = safe_example_hw()

    # 2. Data
    dt = 1e-3
    raw_data = torch.tensor([
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
    ], device=device)

    gwf_input = 0.08 * raw_data
    rf_input = torch.ones(len(gwf_input), device=device)
    rf_input[40:] = -1

    # 3. Run
    pns, res = safe_gwf_to_pns_torch(gwf_input, rf_input, dt, hw)

    gwf_plot = res['gwf']
    pns_plot = res['pns']
    rf_plot  = res['rf']

    t_gwf = torch.arange(len(gwf_plot), device=device) * dt * 1000
    t_pns = torch.arange(len(pns_plot), device=device) * dt * 1000

    gwf_plot = gwf_plot[:-1, :]
    rf_plot = rf_plot[:-1]
    t_gwf = t_gwf[:-1]

    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, figsize=(10,10))

    ax1.plot(t_gwf.cpu(), gwf_plot[:,0].cpu()*1000, label='Gx', color='r')
    ax1.plot(t_gwf.cpu(), gwf_plot[:,1].cpu()*1000, label='Gy', color='g')
    ax1.plot(t_gwf.cpu(), gwf_plot[:,2].cpu()*1000, label='Gz', color='b')
    ax1.set_ylabel('Gradient [mT/m]')
    ax1.set_title('Gradient Waveforms')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    ax1_rf = ax1.twinx()
    ax1_rf.plot(t_gwf.cpu(), rf_plot.cpu(), color='k', linestyle=':', alpha=0.3, label='RF')
    ax1_rf.set_ylabel('RF State')

    ax2.plot(t_pns.cpu(), pns_plot[:,0].cpu(), label='PNS x', color='r', linestyle='--')
    ax2.plot(t_pns.cpu(), pns_plot[:,1].cpu(), label='PNS y', color='g', linestyle='--')
    ax2.plot(t_pns.cpu(), pns_plot[:,2].cpu(), label='PNS z', color='b', linestyle='--')
    ax2.set_ylabel('Stimulation [%]')
    ax2.set_xlabel('Time [ms]')
    ax2.set_title('Peripheral Nerve Stimulation (PNS)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("pns_example_plot.png")
    plt.show()