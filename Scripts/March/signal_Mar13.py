import numpy as np
import pycbc
from pycbc import types, fft, waveform, noise, psd, filter
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.types.timeseries import TimeSeries
import pylab

#------------------------------------------------------------------------------------

def eta(m1,m2):
	return m1*m2 / (m1+m2)**2.0

def chirp_mass(m1,m2):
	return (m1+m2) * eta(m1,m2)**(3.0/5.0)

def chi_s(chi1,chi2):
	return (chi1+chi2) / 2.0

def chi_a(chi1,chi2):
	return (chi1-chi2) / 2.0

def delta(m1,m2):
	return (m1-m2) / (m1+m2)

def dmdt(m1,m2):
	l = 1.4
	dm1dt = -(2.8e-07) * (1.0/m1)**2.0 * (l)**2.0
	dm2dt = -(2.8e-07) * (1.0/m2)**2.0 * (l)**2.0
	return dm1dt + dm2dt

def D_alphaMDR(alpha_MDR, H_0, Omega_M, Omega_Lambda, z):
	return z * (1.0 - z * (3.0*Omega_M/(Omega_M+Omega_Lambda) + 2.0*alpha_MDR) / 4.0 ) / (H_0 * np.sqrt(Omega_M + Omega_Lambda))

# Scalar-Tensor to post-Einsteinian parameters
def ST_to_ppE(m1, m2, chi1, chi2):

	phidot = 1.0
	s1 = (1.0+np.sqrt(1.0-(chi1**2.0)))/2.0
	s2 = (1.0+np.sqrt(1.0-(chi2**2.0)))/2.0
	beta_ppE = -(5.0/1792.0) * phidot**2.0 * eta(m1,m2)**(2.0/5.0) * (m1*s1 - m2*s2)**2.0

	b = -7.0
	return b, beta_ppE


# Gauss-Bonnet to post-Einsteinian parameters
def GB_to_ppE(m1, m2, chi1, chi2, alpha_GB):

	coupling = 16.0*np.pi*alpha_GB**2.0 / ((m1+m2)**4.0)

	BH1_scalar_charge = 4.0*alpha_GB*(np.sqrt(1.0-chi1**2.0)-1.0+chi1**2.0) / (m1**2.0 * chi1**2.0)
	BH2_scalar_charge = 4.0*alpha_GB*(np.sqrt(1.0-chi2**2.0)-1.0+chi2**2.0) / (m2**2.0 * chi2**2.0)

	s1 = BH1_scalar_charge*m1**2.0 / (2.0*alpha_GB)
	s2 = BH2_scalar_charge*m2**2.0 / (2.0*alpha_GB)
	beta_ppE = -(5.0/7168.0)*coupling*(m1**2.0 * s2 - m2**2.0 * s1)**2.0 / ((m1+m2)**4.0 * eta(m1,m2)**(18.0/5.0))

	b = -7.0
	return b, beta_ppE

# dynamical Chern-Simons to post-Einsteinian parameters
def dCS_to_ppE(m1, m2, chi1, chi2, alpha_CS):

	coupling = 16.0*np.pi*alpha_CS**2.0 / ((m1+m2)**4.0)

	beta_ppE = (1549225.0/11812864.0) * (coupling / eta(m1,m2)**(14.0/5.0)) * ((1.0 - (231808.0/61969.0)*eta(m1,m2))*chi_s(chi1,chi2)**2.0 + (1.0 - (16068.0/61969.0)*eta(m1,m2))*chi_a(chi1,chi2)**2.0 - 2.0*delta(m1,m2)*chi_s(chi1,chi2)*chi_a(chi1,chi2))

	b = -1.0
	return b, beta_ppE

# Einstein-Aether to post-Einsteinian parameters
def EA_to_ppE(c_1, c_2, c_3, c_4):

	c_14 = c_1 + c_4
	c_123 = c_1 + c_2 + c_3
	c_plus = c_1 + c_3
	c_minus = c_1 - c_3

	w_0 = np.sqrt((2.0 - c_14) * c_123 / ((2.0 + 3.0*c_2 + c_plus) * (1.0 - c_plus) * c_14))
	w_1 = np.sqrt((2.0*c_1 - c_plus*c_minus) / (2.0*(1.0 - c_plus)*c_14))
	w_2 = np.sqrt(1.0 / (1.0 - c_plus))

	beta_ppE = -(3.0/128.0) * ( (1.0 - c_14/2.0) * (1.0/w_2 + 2.0 * c_14 * c_plus**2.0 / ((c_plus + c_minus - c_minus*c_plus)**2.0 * w_1) + 3.0*c_14 / (2.0 * w_0 * (2.0 - c_14))) - 1.0 )

	b = -5.0
	return b, beta_ppE

# Khronometric Gravity to post-Einsteinian
def khronometric_to_ppE(alpha_KG, beta_KG, lambda_KG):

	w_0 = np.sqrt((alpha_KG - 2.0) * (beta_KG + lambda_KG) / (alpha_KG * (beta_KG - 1.0) * (2.0 + beta_KG + 3.0*lambda_KG)))
	w_2 = np.sqrt(1.0 / (1.0 - beta_KG))

	beta_ppE = -(3.0/128.0) * ((1.0 - beta_KG) * (3.0*beta_KG / (2.0*w_0*w_2*(1.0-beta_KG))) - 1.0)

	b = -5.0
	return b, beta_ppE

# extra dimension
def extra_dimension_to_ppE(m1, m2):

	beta_ppE = (25.0/851968.0) * dmdt(m1,m2) * (3.0 - 26.0*eta(m1,m2) + 34.0*eta(m1,m2)**2.0) / (eta(m1,m2)**(2.0/5.0) * (1.0 - 2.0*eta(m1,m2)))
	
	b = -13.0
	return b, beta_ppE

# varying G
def varying_G_to_ppE(m1, m2, dGdt):

	beta_ppE = -(25.0/65536.0) * dGdt * chirp_mass(m1,m2)

	b = -13.0
	return b, beta_ppE

# modified dispersion relation
def MDR_to_ppE(m1, m2, chi1, chi2, alpha_MDR, z):
	
	beta_ppE = (np.pi**(2.0 - alpha_MDR) / (1.0 - alpha_MDR)) * (D_alphaMDR(alpha_MDR,z) / (lambda_A**(2.0 - alpha_MDR))) * (chirp_mass(m1,m2)**(1.0 - alpha_MDR) / ((1.0 + z)**(1.0 - alpha_MDR)))
	
	b = 3.0 * (alpha_MDR - 1.0)
	return b, beta_ppE
	

def ppE_to_h(m1,m2,b,beta,freq_sp,freq_sc,sp,sc):
	vf_sp = np.zeros((len(freq_sp)))
	vf_sc = np.zeros((len(freq_sc)))
	ppE_factor_sp = np.zeros((len(freq_sp)),dtype=np.complex_)
	ppE_factor_sc = np.zeros((len(freq_sc)),dtype=np.complex_)
	hp = np.zeros((len(freq_sp)),dtype=np.complex_)
	hc = np.zeros((len(freq_sc)),dtype=np.complex_)

	vf_sp = (np.pi*(m1+m2)*freq_sp)**(1.0/3.0)
        vf_sc = (np.pi*(m1+m2)*freq_sc)**(1.0/3.0)

	ppE_factor_sp[0] = 1.0
	ppE_factor_sc[0] = 1.0
        ppE_factor_sp[1:] = np.exp( (np.full(((len(freq_sp)-1)),beta)) * (np.array(vf_sp)[1:]**b) * (np.full(((len(freq_sp)-1)),1j,dtype=np.complex_)) )
        ppE_factor_sc[1:] = np.exp( (np.full(((len(freq_sc)-1)),beta)) * (np.array(vf_sc)[1:]**b) * (np.full(((len(freq_sc)-1)),1j,dtype=np.complex_)) )


	hp = sp * ppE_factor_sp
	hc = sc * ppE_factor_sc
	return hp, hc


def IFFT_to_TD(hp,hc):
	tp = np.fft.ifft(hp)
	tc = np.fft.ifft(hc)
	return tp, tc

def FFT_to_FD(tp,tc):
	fp = np.fft.fft(tp)
	fc = np.fft.fft(tc)
	return fp, fc

def Noise(flow,delta_f,delta_t,tlen):
	flen = int(1.0/delta_t/delta_f) / 2 + 1
	p_s_d = psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
	Nt = int(tlen/delta_t)
	return noise.noise_from_psd(Nt, delta_t, p_s_d, seed=127)

def Array_Match(tp,tc,noise):
	mid_index_A = int(np.ceil((len(noise)-1) / 2.0))
	mid_index_B = mid_index_A + len(tp)
	tp_long = np.zeros((len(noise)),dtype=np.complex_)
	tc_long = np.zeros((len(noise)),dtype=np.complex_)

	tp_long[mid_index_A:mid_index_B] = tp[:]
	tc_long[mid_index_A:mid_index_B] = tc[:]
	return tp_long, tc_long
	

def Matched_Filter(template_p,template_c,data_p,data_c,flen,delta_f,flow):
	p_s_d = psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
	SNRp = filter.matched_filter(template_p, data_p, psd=p_s_d, 
                                      low_frequency_cutoff=flow)
	SNRc = filter.matched_filter(template_c, data_c, psd=p_s_d, 
                                      low_frequency_cutoff=flow)
	return SNRp,SNRc

#---------------------------------------------------------------------------------

# Test Run

m1 = 25.0
m2 = 25.0
chi1 = 0.5
chi2 = 0.3
alpha_GB = 1.0
alpha_CS = 1.0

flow = 20.0
sample_rate = 4096.0
tlen = 128
N_t = sample_rate * tlen
flen = int(N_t / 2.0) + 1
delta_t = 1.0 / sample_rate
delta_f = 1.0 / tlen

sp, sc = waveform.get_fd_waveform(approximant='IMRPhenomD', mass1=m1, mass2=m2,
                                  delta_f=delta_f, f_lower=flow)

sp_array = np.array(sp.data)
sc_array = np.array(sc.data)
freq_sp = np.array(sp.sample_frequencies.data)
freq_sc = np.array(sc.sample_frequencies.data)


#sp, sc = waveform.get_fd_waveform(approximant='SEOBNRv4_ROM', mass1=m1, mass2=m2,
#                                  delta_f=delta_f, f_lower=flow)

#sp, sc = waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=m1, mass2=m2,
#                                  delta_f=delta_f, f_lower=flow)

# Scalar-Tensor Theory
#b, beta = ST_to_ppE(m1,m2,chi1,chi2)

# Gauss-Bonnet Theory
b, beta = GB_to_ppE(m1,m2,chi1,chi2,alpha_GB)

# Chern-Simons Theory
#b, beta = dCS_to_ppE(m1,m2,chi1,chi2,alpha_CS)

hp, hc = ppE_to_h(m1,m2,b,beta,freq_sp,freq_sc,sp_array,sc_array)
tp, tc = IFFT_to_TD(hp,hc)

noise = np.array(Noise(flow,delta_f,delta_t,tlen))
tp_long, tc_long = Array_Match(tp,tc,noise)

signal_p = noise + tp_long
signal_c = noise + tc_long
signal_pTS = TimeSeries(np.real(signal_p), delta_t = delta_t, dtype = noise.dtype)
signal_cTS = TimeSeries(np.real(signal_c), delta_t = delta_t, dtype = noise.dtype)

template_pFS = FrequencySeries(np.zeros(len(noise)/2 + 1),delta_f = delta_f, dtype = np.complex_)
template_cFS = FrequencySeries(np.zeros(len(noise)/2 + 1),delta_f = delta_f, dtype = np.complex_)
template_pFS[:len(sp)] = sp
template_cFS[:len(sc)] = sc

noise_TS = TimeSeries(noise, delta_t = delta_t, dtype = noise.dtype)

Signal_OFF = Matched_Filter(template_pFS,template_cFS,noise_TS,noise_TS,flen,delta_f,flow)
Signal_ON = Matched_Filter(template_pFS,template_cFS,signal_pTS,signal_cTS,flen,delta_f,flow)

