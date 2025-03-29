# -*- coding: utf-8 -*-
"""
Calculate 90 CL limits and sensitivity bands for LBC data patterns.
ALS, February 2025.
"""

import sys
import math
import numpy as np
import pandas as pd
import csv
import os
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# DARK CURRENT BACKGROUND
# 11 21 111 31 211 22
#Bkg_dark    = np.array([141.4,4.24e-2,4.24e-2,7.7e-6,2.2e-5,3.6e-6])

Bkg_dark    = np.array([141.4,4.24e-2,4.24e-2,7.7e-6,2.2e-5,3.6e-6])
Bkg_p = np.array([0.039,0.039,0.016,0.052,0.035,0.011])
Data0  = np.array([144,0,0,1,0,0])  # 31:0

# RATES: RADIOACTIVE DECAYS BACKGROUND + PATTERNS
fname = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Rate files\rate_file_patterns_noise_comp_masses_new_efficiencies_neh_11.csv"
df = pd.read_csv(fname)

mX_eV_arr = df["mX_eV"].unique()

# PREPARE SIGNAL AND BACKGROUND
def prepareSignal(ip):
    mX_eV = mX_eV_arr[ip]
    signal_data = df[df["mX_eV"] == mX_eV].rate_pattern.to_numpy()
    #background_data = df[df["mX_eV"] == mX_eV].rate_bkg.to_numpy()
    
    # Apply normalization
    Signal = signal_data * 1.257 * 1000. * 1.e-35
    #Bkg = Bkg_dark + (background_data * 1.257 * 1000.)
    Bkg = Bkg_dark + Bkg_p
    
    return Signal, Bkg

# EXCLUSION LIMITS CALCULATION
def muHat(x):
    val = sum(data[i] * Sref[i] / (x * Sref[i] + Bkg[i]) - Sref[i] for i in range(A, B))
    return val 

def T_mu(n, mu, s, b):
    nu = mu * s + b
    nu_h = Mu_hat * s + b
    if Mu_hat > mu:
        return 0
    if Mu_hat <= 0:
        nu_h = b
    if nu_h == 0:
        return 2 * nu
    return -2 * (n * math.log(nu) - nu - (n * math.log(nu_h) - nu_h))

def Prob(muS):
    tmu_values = np.zeros(N)
    tData = sum(T_mu(data[i], muS, Sref[i], Bkg[i]) for i in range(A, B) if Sref[i] > 0)
    for i in range(A, B):
        if Sref[i] > 0:
            poisson_samples = np.random.poisson(muS * Sref[i] + Bkg[i], N)
            tmu_values += [T_mu(n, muS, Sref[i], Bkg[i]) for n in poisson_samples]
    count = sum(1 for x in tmu_values if x >= tData)
    return round((count - N * CL) / math.sqrt(N * CL * (1 - CL)) * 3) * math.sqrt(N * CL * (1 - CL)) * 3

# Simulation Parameters
N = 10000
Npattern = 6
CL = 0.1
A = 0
B = Npattern
data = Data0
xMass_list = []
xLim_list = []

for ip in range(0, 25):
    Sref, Bkg = prepareSignal(ip)
    if muHat(0) < 0:
        Mu_hat = 0
        muLim = 10 * data[0] / Sref[0]
    else:
        Mu_hat = brentq(muHat, 0, 10 * data[0] / Sref[0])
        muLim = 100 * Mu_hat    
        
    root = brentq(Prob, 0, muLim)
    xMass_list.append(mX_eV_arr[ip] / 1E6)
    xLim_list.append(root * 10**(-35))
    print(f"Limit for {mX_eV_arr[ip] / 1E6:.2e} MeV calculated")

print("-------------------------------------------------------------")    

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(xMass_list, xLim_list, color='black', lw=2, label='DAMIC-M, This work')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('DM mass (MeV)')
plt.ylabel(r'Cross section (cm$^2$)')
plt.title('90%CL Upper limit on DM-electron cross section dRdQ_massless')
plt.legend()
plt.show()

# Definir la ruta del archivo
save_path = r"C:\Users\Asus\Documents\MÁSTER\TFM\QEdark\Exclusion Limits\exclusion_limits_new_eff.txt"

# Nombre del nuevo header de la columna
new_column_header = "Limit (cm^2) neh = 11"

# Comprobar si el archivo ya existe
if os.path.exists(save_path):
    # Leer el archivo existente
    file_exc = pd.read_csv(save_path, delim_whitespace=True)
else:
    # Si el archivo no existe, creamos un DataFrame vacío con la primera columna de masas
    file_exc = pd.DataFrame(columns=["DM Mass (MeV)"])

# Convertir listas en un DataFrame
new_data = pd.DataFrame({"DM Mass (MeV)": xMass_list, new_column_header: xLim_list})

# Verificar si "DM Mass (MeV)" existe en los datos previos
if "DM Mass (MeV)" in file_exc.columns:
    # Unir con los datos anteriores sin perder las columnas existentes
    file_exc = pd.merge(file_exc, new_data, on="DM Mass (MeV)", how="outer")
else:
    # Si el archivo estaba vacío, solo usamos los nuevos datos
    file_exc = new_data

# Guardar el archivo con la nueva columna agregada
file_exc.to_csv(save_path, sep=" ", index=False)

print(f"File saved in: {save_path}")