import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Dados do osciloscópio na configuração 1
df = pd.read_csv('./CH1.txt', sep='\t', names=['t', 'VC'])
df['VA'] = pd.read_csv('./CH0.txt', sep='\t', names=['t', 'VA'])['VA']

pd.set_option('display.max_rows', None)
print(df)

def voltage_increasing(t, v0, tau):
    return v0 * (1 - np.exp(-t/tau))

def voltage_decreasing(t, v0, tau):
    return v0 * np.exp(-t/tau)

time_values = [np.array(df['t'][:164]), np.array(df['t'][163:327]), np.array(df['t'][326:])]
y_values = [np.array(df['VC'][:164]), np.array(df['VC'][163:327]), np.array(df['VC'][326:])]

plt.plot(df['t'], df['VC'], label='VC(t) Experimental')

popt, pcov = curve_fit(voltage_increasing, time_values[0] - df['t'][0], y_values[0])
v0_opt, tau_opt = popt

plt.plot(time_values[0], voltage_increasing(time_values[0] - df['t'][0], v0_opt, tau_opt), color='tab:orange', label='VC(t) Fit')
plt.plot(time_values[1], voltage_decreasing(time_values[1] - df['t'][163], v0_opt, tau_opt), color='tab:orange')
plt.plot(time_values[2], voltage_increasing(time_values[2] - df['t'][326], v0_opt, tau_opt), color='tab:orange')

# Tensão no resistor
df['VR'] = df['VA'] - df['VC']
plt.plot(df['t'], df['VR'], color='tab:green', label='VR(t)')

plt.title('10k-100nF')
plt.xlabel('A0')
plt.ylabel('A1')
plt.legend()
plt.show()
