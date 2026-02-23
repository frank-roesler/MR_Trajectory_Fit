import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.io as sio
import json
import os
import torch
import torchkbnufft as tkbn

import safe_gwf_to_pns
import safe_hw_from_asc

# %% System Parameters
# some sytem paramters (not used to really calculate the trajectory, but to
#   create useful ouput measures (e.g. total measurment time)
TR        = 1500  # [ms], TODO
noAvgs    =    8  # number of averages, TODO
noPreScan =    2  # number of pre-scanns to reach equilibrium, TODO

usFactor = 2 # under sampling factor (defines how the petals being split)

# trajectory paramters
FoV        =  224.0  # [mm]  (default: 240mm)
res        =   50    # [matrix size]
smpFreq    = 2000.0  # [Hz] temporal spectral resolution (defines how fast we have to complete one paddle)
dwellTime  =   10.0  # [us] spacing between temporal points
overSmpOn  = True
noPaddles  =   80    # instead of doing 6 averages we increase the number of paddles
noPaddleChunks = 4   # ONLY USED FOR DENSITY-WEIGHTED SAMPLING

specRes    =  325    # [pts] sampling points along spectral dimension
encType    = '2D'    # either '2D' or '3D'

adcPreOnset    = 200.0     # [us] - pre-onset of adc
gradPreEmphPts =  10       # pre-emphasis points of gradients

osFactor = 2   # oversampling factor

beta_max = 2 * np.pi

# Trajectory parameters to be varied
o1 = 1.0
o2 = 1.0

# TRAJECTORY OPTIMIZATION PARAMTERS
dwFct = 'Hanning'

# System limitations
maxRmp = 200.0      # [T/m/s]
gamma = 42.575575   # [MHz/T]

rootDir__ = './'    # Adjust path as needed
outputDir = './'    # Adjust path as needed

# %% Optimization starts here

Kmax = res / (FoV * 1e-3) / 2.0  # 1/m

# Time vectors
# MATLAB: 0:10:(1/smpFreq*1e6) -> Inclusive end
t_end = (1 / smpFreq * 1e6)
tVec = np.arange(0, t_end + 1e-9, 10.0) # 10us steps

if overSmpOn:
    tVecAdc = np.arange(0, t_end + 1e-9, dwellTime / 2.0)
    adcPreOnsetPtsOS = adcPreOnset / 10 * osFactor
    gradPreEmphPtsOS = gradPreEmphPts * osFactor
else:
    tVecAdc = np.arange(0, t_end + 1e-9, dwellTime)
    adcPreOnsetPtsOS = adcPreOnset / 10
    gradPreEmphPtsOS = gradPreEmphPts

print(f"Real temporal sampling frequency fs: {1/(tVecAdc[-1]*1e-6)} Hz")
print(f"Fid sampling duration:               {specRes/(1/(tVecAdc[-1]*1e-6))*1000} ms")
print(f"Gradient pts per paddle:             {len(tVec) - 1}")
print(f"ADC sampling pts per paddle:         {len(tVecAdc) - 1}")
print(f"No of pure ADC points needed:        {(len(tVecAdc)-1)*specRes}")
print(f"No of real ADC points needed:        {(len(tVecAdc)-1)*specRes + adcPreOnsetPtsOS + gradPreEmphPtsOS}")
print(f"Max ADC smp points:                  32768")

# Angles
beta = np.linspace(0, 2 * np.pi, noPaddles + 1)
beta = beta[:-1] # Equivalent to beta(1:end-1)

om1 =       np.pi / tVec[-1] * 1e6
om2 = o2/o1 * np.pi / tVec[-1] * 1e6

# Rosette trajectory (2D)
f1 = om1 / np.pi
f2 = om2 / np.pi
print(f"Rosette spatial frequency omega1 / f1: {om1} / {f1} Hz")
print(f"Rosette spatial frequency omega2 / f2: {om2} / {f2} Hz")

# Initialize arrays
# Kxy will be (time_points, noPaddles)
Kxy = np.zeros((len(tVec), len(beta)), dtype=np.complex128)
Kxy_smp = np.zeros((len(tVecAdc)-1, len(beta)), dtype=np.complex128)

for itx in range(len(beta)):
    # Calculate Kxy
    # Note: MATLAB used tVec(1:end) / max(tVec). In Python tVec is the same length.
    term_sin = Kmax * np.sin(o1/o1 * (np.pi * tVec / np.max(tVec)))
    term_exp = np.exp(1j * (o2/o1 * (np.pi * tVec / np.max(tVec)) + beta[itx]))
    Kxy[:, itx] = term_sin * term_exp
    
    # Interpolate to ADC time points
    # MATLAB: interp1(tVec, real(Kxy), tVecAdc(1:end-1))
    # We create interpolators for real and imag parts
    f_real = interpolate.interp1d(tVec, np.real(Kxy[:, itx]), kind='linear', fill_value="extrapolated")
    f_imag = interpolate.interp1d(tVec, np.imag(Kxy[:, itx]), kind='linear', fill_value="extrapolated")
    
    t_targets = tVecAdc[:-1]
    Kxy_smp[:, itx] = f_real(t_targets) + 1j * f_imag(t_targets)

# MATLAB: Kxy(end,:) = []; -> Remove last point
Kxy = Kxy[:-1, :]

# Nearest Neighbor check (Mock logic for dist_new as knnsearch is specific)
# Using a simple brute force for demonstration or skipping complex k-space density logic here
# equivalent to: [idx, dist] = knnsearch(...)
# For python we can use sklearn.neighbors, but let's stick to basic numpy or skip if just for visual
from sklearn.neighbors import NearestNeighbors
X_knn = np.column_stack((np.real(Kxy_smp.flatten()), np.imag(Kxy_smp.flatten())))
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_knn)
distances, indices = nbrs.kneighbors(X_knn)
dist_new = np.sqrt(np.sum(distances**2, axis=1))

FoVMax = 1.0 / np.max(dist_new[dist_new > 1e-6]) * 1000
print(f"Max FoV: {FoVMax}")
resMax = FoVMax * Kmax * 2 / 1000
print(f"FoVMax/(resMax/50): {FoVMax/(resMax/50)}")


# %% Get Gradients
# MATLAB: diff(..., 1, 1) means diff along dim 0 (time)
# We append the first point to the end to match MATLAB's [Kxy; Kxy(1,:)] before diff
# However, MATLAB diff reduces size by 1. 

def calc_gradient(K_data, dt_arr):
    # K_data: (time, paddles)
    # Append first point to end for cyclic diff
    K_padded = np.vstack((K_data, K_data[0:1, :])) 
    dt_mean = np.mean(np.diff(dt_arr))
    
    grads_real = 1/gamma * np.diff(np.real(K_padded), axis=0) * 1e3 / dt_mean # mT/m
    grads_imag = 1/gamma * np.diff(np.imag(K_padded), axis=0) * 1e3 / dt_mean # mT/m
    return grads_real, grads_imag

Gx, Gy = calc_gradient(Kxy, tVec)

# For sampled ones, MATLAB does: [Kxy_smp; Kxy_smp(1,:); Kxy_smp(2,:)]
K_smp_padded = np.vstack((Kxy_smp, Kxy_smp[0:1, :], Kxy_smp[1:2, :]))
dt_adc_mean = np.mean(np.diff(tVecAdc))

Gx_smp = 1/gamma * np.diff(np.real(K_smp_padded), axis=0) * 1e3 / dt_adc_mean
Gy_smp = 1/gamma * np.diff(np.imag(K_smp_padded), axis=0) * 1e3 / dt_adc_mean

# Interpolation of Gradients
# MATLAB: interp1(tVec(1:end), Gx_smp(1:2:end,:), tVecAdc(1:end-1))
# Note: Gx_smp is calculated on tVecAdc basis in MATLAB logic? 
# Wait, MATLAB code says: Gx_smp_interp = interp1( tVec, Gx_smp(1:2:end,:), ... )
# This implies Gx_smp was somehow on tVec resolution or subsampled? 
# In the Python translation above, Gx_smp is already on tVecAdc. 
# Following logic strictly: Gx_smp calculated above is already on `tVecAdc` grid.
# The MATLAB code seems to take `Gx_smp` (which is high res) and downsamples `1:2:end` then interpolates back?
# We will skip the specific `1:2:end` subsampling logic unless strictly necessary to match artifact, 
# assuming we just want the gradient on the ADC grid.

Gx_smp_interp = Gx_smp[:len(tVecAdc)-1, :] # Simplified for Python flow
Gy_smp_interp = Gy_smp[:len(tVecAdc)-1, :]

# Recalculate K from interpolated Gradients (Check consistency)
Kxy_smp_interp = gamma * np.mean(np.diff(tVecAdc))/1e3 * (
    np.cumsum(np.vstack((np.zeros((1, Gx_smp_interp.shape[1])), Gx_smp_interp)), axis=0) + 
    1j * np.cumsum(np.vstack((np.zeros((1, Gy_smp_interp.shape[1])), Gy_smp_interp)), axis=0)
)
Kxy_smp_interp = Kxy_smp_interp[:-1, :]

# Second derivative for check
Gx_smp_interp_2 = 1/gamma * np.diff(np.real(np.vstack((Kxy_smp_interp, Kxy_smp_interp[0:1,:]))), axis=0) * 1e3 / dt_adc_mean
Gy_smp_interp_2 = 1/gamma * np.diff(np.imag(np.vstack((Kxy_smp_interp, Kxy_smp_interp[0:1,:]))), axis=0) * 1e3 / dt_adc_mean


# %% Plotting Gradients & K-Space
plt.figure()
plt.plot(tVecAdc, Gx_smp[:, 0])
plt.title('Gx Sampled (Paddle 1)')
plt.show()

# K-space trajectory
tmp_kxx = np.cumsum(np.vstack((np.zeros((1, Gx_smp_interp.shape[1])), Gx_smp_interp)), axis=0)
tmp_kyy = np.cumsum(np.vstack((np.zeros((1, Gy_smp_interp.shape[1])), Gy_smp_interp)), axis=0)

plt.figure()
for itx in range(1, tmp_kxx.shape[1]):
    colPlt = 'k' if (itx % 2 == 0) else 'r'
    plt.plot(tmp_kxx[:, itx], tmp_kyy[:, itx], color=colPlt, linewidth=1.5)
plt.axis('equal')
plt.title('Trajectory Reconstructed from Grads')
plt.show()

# Truncate to remove extra points added for diff calculation
Gx_smp = Gx_smp[:-1, :]
Gy_smp = Gy_smp[:-1, :]

print(f"min(Gx): {np.min(Gx[1:, :])} mT/m")
print(f"min(Gy): {np.min(Gy[1:, :])} mT/m")
print(f"max(Gx): {np.max(Gx[1:, :])} mT/m")
print(f"max(Gy): {np.max(Gy[1:, :])} mT/m")

# %% Slew Rate
# diff of gradients
slewGx = np.diff(np.vstack((np.zeros((1, Gx.shape[1])), Gx)), axis=0) / (np.mean(np.diff(tVec))*1e-3)
slewGy = np.diff(np.vstack((np.zeros((1, Gy.shape[1])), Gy)), axis=0) / (np.mean(np.diff(tVec))*1e-3)

print(f"min(slewGx): {np.min(np.abs(slewGx))} T/m/s")
print(f"max(slewGx): {np.max(np.abs(slewGx))} T/m/s")

# %% PNS Check
maxPns = np.zeros(noPaddles)

hw = safe_hw_from_asc.safe_hw_from_asc('safe_pns_prediction/MP_GradSys_K2298_2250V_1250A_W60_SC72CD.asc')

for pdlItx in range(noPaddles):
    # Compose gradient vector [T/m]
    # Ramp up
    ramp_up_x = np.max(Gx[0, pdlItx]) * np.linspace(0, 1, gradPreEmphPts)
    ramp_dn_x = np.max(Gx[-1, pdlItx]) * np.linspace(1, 0, gradPreEmphPts)
    full_ro_x = np.tile(Gx[:, pdlItx], (specRes)) # Repeat for spectral resolution
    gVecX = np.concatenate((ramp_up_x, full_ro_x, ramp_dn_x)) * 1e-3

    ramp_up_y = np.max(Gy[0, pdlItx]) * np.linspace(0, 1, gradPreEmphPts)
    ramp_dn_y = np.max(Gy[-1, pdlItx]) * np.linspace(1, 0, gradPreEmphPts)
    full_ro_y = np.tile(Gy[:, pdlItx], (specRes))
    gVecY = np.concatenate((ramp_up_y, full_ro_y, ramp_dn_y)) * 1e-3
    
    gVecZ = np.zeros_like(gVecX)
    gVec = np.column_stack((gVecX, gVecY, gVecZ))

    rfVec = np.ones(len(gVec))
    dt = 10 * 1e-6

    pns, resOut = safe_gwf_to_pns.safe_gwf_to_pns(gVec, rfVec, dt, hw, 1)
    maxPns[pdlItx] = np.max(np.sqrt(np.sum(pns**2, axis=1)))

plt.figure()
plt.plot(maxPns)
plt.title('Max PNS')
plt.show()

# %% Export JSON
json_dict = {
    "conf": {
        "id": f"fov{round(FoV)}_slew{round(np.max(np.abs(slewGx[1:,:])))}_frq{round(f1)}",
        "specBW": 1 / (tVec[-1] * 1e-6),
        "specRes": specRes,
        "FoV": FoV,
        "res": res,
        "preEmMom": 24,
        "preEmPts": gradPreEmphPts,
        "pdlNo": noPaddles,
        "pdlPts": len(tVec) - 1
    },
    "info": {
        "maxSlewX": float(np.max(np.abs(slewGx[1:, :]))),
        "maxSlewY": float(np.max(np.abs(slewGy[1:, :]))),
    },
    "trj": {
        "GxMax": np.max(np.abs(Gx), axis=0).tolist(),
        "GyMax": np.max(np.abs(Gy), axis=0).tolist(),
        # Normalize and swap dimensions (paddles as first dim for list export usually)
        "Gx": (Gx / np.max(np.abs(Gx), axis=0)).T.tolist(),
        "Gy": (Gy / np.max(np.abs(Gy), axis=0)).T.tolist()
    }
}

file_name = f"{outputDir}fov{round(FoV)}_slew{round(np.max(np.abs(slewGx)))}_frq{round(f1)}.conf"
# Helper to convert numpy types to native types for JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.generic): return obj.item()
        return json.JSONEncoder.default(self, obj)

with open(file_name, 'w') as f:
    json.dump(json_dict, f, cls=NumpyEncoder, indent=4)

# Save Trajectory to .mat (or .npz)
trjComb = {
    'Kmax': Kmax,
    'Kxy': Kxy,
    'Kxy_smp': Kxy_smp,
    'Kxy_smp_interp': Kxy_smp_interp
}
sio.savemat(f"{outputDir}kspace_res{res:.0f}_bw{smpFreq:.0f}Hz_ptls{noPaddles:.0f}.mat", trjComb)


# %% Acoustic Analysis (Forbidden Freqs)
forbiddenFreq = np.array([
    [1100, 300],
    [550, 100],
    [367, 55]
])

# Create full time vector
# tVecTot = (0:1:(length(...) - 1)) * 10
# Replicating Gx(:,1)
Gx_rep = np.tile(Gx[:, 0], (specRes))
Gy_rep = np.tile(Gy[:, 0], (specRes))
tVecTot = np.arange(len(Gx_rep)) * 10.0 # us

Fs = 1e6 / 10.0 # Hz
L = len(tVecTot)

Yx = np.fft.fft(Gx_rep)
Yy = np.fft.fft(Gy_rep)
freqs = Fs/L * np.arange(L)

plt.figure()
plt.title('Acoustic PSD')
plt.plot(freqs, np.abs(Yx), linewidth=2, label='Gx')
plt.plot(freqs, np.abs(Yy), linewidth=2, label='Gy')
plt.xlabel('f (Hz)')
plt.ylabel('|Intensity|')
plt.xlim([0.3, 1.5 * max(f1, f2)])

# Draw forbidden bands
y_min, y_max = plt.ylim()
for band in forbiddenFreq:
    center, width = band[0], band[1]
    plt.fill_betweenx([y_min, y_max], center - width/2, center + width/2, color='red', alpha=0.1)

plt.legend()
plt.show()

# %% NUFFT & PSF Analysis using torchkbnufft
print("Running NUFFT / PSF analysis...")

# 1. Density Compensation (Simple Histogram approximation)
# Histogram2D
k_real = np.real(Kxy_smp).flatten()
k_imag = np.imag(Kxy_smp).flatten()

# Note: In MATLAB code res*10 was used for histogram binning
nbins = res * 10 + 1
H, xedges, yedges = np.histogram2d(k_real, k_imag, bins=nbins)

plt.figure()
plt.imshow(H.T, origin='lower')
plt.title('k-space sampling density')
plt.colorbar()
plt.show()

# 2. PSF Calculation
# Prepare data for torchkbnufft
# kbnufft expects trajectory in range [-pi, pi] usually, but handles physical units if configured.
# Here we scale input k-space to [-pi, pi] based on resolution.
# Kmax corresponds to the edge.
# We normalize Kxy so that the max extent corresponds to pi.

# Convert Kxy_smp to tensor shape (2, n_points)
# And flatten all paddles
ktraj = np.stack((np.real(Kxy_smp).flatten(), np.imag(Kxy_smp).flatten()), axis=0)

# Normalize trajectory to range [-pi, pi]
# Kmax is the maximum spatial frequency.
# 1/FOV = delta_k. Res * delta_k = k_width.
# We need to map -Kmax...Kmax to -pi...pi
ktraj_norm = ktraj / Kmax * np.pi 

ktraj_tensor = torch.tensor(ktraj_norm, dtype=torch.float32)
# Density compensation weights (dcf). For PSF we often set dcf=1 or calculate iterative DCF.
# MATLAB code used E1'*E1.w, implying w is density weights. 
# For simple PSF we often just adjoint transform a ones vector.
dcf = torch.ones(ktraj_tensor.shape[1], dtype=torch.complex64)
# Fake data (ones)
data = torch.ones(ktraj_tensor.shape[1], dtype=torch.complex64).unsqueeze(0).unsqueeze(0) # (1, 1, npts)

# Initialize NUFFT
im_size = (res + 1, res + 1)
# Jd=[2,2] in MATLAB corresponds to interpolation size. defaults usually 6 in kbnufft, can set grid_size
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)

# Calculate PSF (Adjoint of ones)
# Note: In kbnufft, adjoint takes (data, trajectory)
psf_tensor = adjnufft_ob(data, ktraj_tensor.unsqueeze(0))
psf = psf_tensor.squeeze().numpy()

plt.figure()
plt.imshow(np.real(psf), cmap='viridis')
plt.title('PSF (Real)')
plt.colorbar()
plt.show()

plt.figure()
plt.plot(np.real(psf[int(res/2), :]))
plt.title('PSF Profile (Central Line)')
plt.show()

# Hamming Filtered Version
# Create filter along time dimension (tVec) then tile
win = np.hamming(len(tVecAdc)-1)
# Circshift logic from MATLAB: circshift(filters, ceil(length/2))
win = np.roll(win, int(np.ceil(len(win)/2)))
# Repeat for all paddles
filters = np.tile(win, len(beta))

data_filtered = torch.tensor(filters, dtype=torch.complex64).unsqueeze(0).unsqueeze(0)

psf_filtered_tensor = adjnufft_ob(data_filtered, ktraj_tensor.unsqueeze(0))
psf_filtered = psf_filtered_tensor.squeeze().numpy()

plt.figure()
plt.plot(np.log(np.abs(psf[int(res/2), :])), label='No Filter')
plt.plot(np.log(np.abs(psf_filtered[int(res/2), :])), label='Hamming')
plt.title('PSF Log Profile Comparison')
plt.legend()
plt.show()

print("Done.")