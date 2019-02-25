import numpy as np
from pycbc import types, fft, waveform, noise, psd

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
        ppE_factor_sp = np.exp( (np.full((len(freq_sp)),beta)) * (np.array(vf_sp)**b) * (np.full((len(freq_sp)),1j,dtype=np.complex_)) )
        ppE_factor_sc = np.exp( (np.full((len(freq_sc)),beta)) * (np.array(vf_sc)**b) * (np.full((len(freq_sc)),1j,dtype=np.complex_)) )


        hp = sp * ppE_factor_sp
        hc = sc * ppE_factor_sc
        return hp, hc

def IFFT_to_TD(m1,m2,b,beta,freq_sp,freq_sc,sp,sc,delta_t,delta_f):
        hp = ppE_to_h(m1,m2,b,beta,freq_sp,freq_sc,sp,sc)[0]
        hc = ppE_to_h(m1,m2,b,beta,freq_sp,freq_sc,sp,sc)[1]

        tlen_sp = int(1.0/delta_t/delta_f)
        tlen_sc = int(1.0/delta_t/delta_f)
#        hp.resize(tlen_sp/2 + 1)
#        hc.resize(tlen_sc/2 + 1)

        tp = np.fft.ifft(hp)
        tc = np.fft.ifft(hc)
        return tp, tc

def Noise(flow,delta_f,delta_t):
        flen = int(2048/delta_f) + 1
        p_s_d = psd.aLIGOZeroDetHighPower(flen, delta_f, flow)
        t_samples = int(1.0/delta_t/delta_f)
        return noise.noise_from_psd(t_samples, delta_t, p_s_d, seed=127)

#---------------------------------------------------------------------------------

# Test Run

m1 = 5.0
m2 = 5.0
chi1 = 0.5
chi2 = 0.3
alpha_GB = 1.0
alpha_CS = 1.0

sp, sc = waveform.get_fd_waveform(approximant='IMRPhenomD', mass1=5, mass2=5,
                                  delta_f=1.0/4, f_lower=40)

sp_array = np.array(sp.data)
sp_array = np.delete(sp_array,0)
sc_array = np.array(sc.data)
sc_array = np.delete(sc_array,0)
freq_sp = np.array(sp.sample_frequencies.data)
freq_sp = np.delete(freq_sp,0)
freq_sc = np.array(sc.sample_frequencies.data)
freq_sc = np.delete(freq_sc,0)


#sp, sc = waveform.get_fd_waveform(approximant='SEOBNRv4_ROM', mass1=5, mass2=5,
#                                  delta_f=1.0/4, f_lower=40)

#sp, sc = waveform.get_fd_waveform(approximant='IMRPhenomPv2', mass1=5, mass2=5,
#                                  delta_f=1.0/4, f_lower=40)

# Scalar-Tensor Theory
#b = ST_to_ppE(m1,m2,chi1,chi2)[0]
#beta = ST_to_ppE(m1, m2, chi1, chi2)[1]

# Gauss-Bonnet Theory
b = GB_to_ppE(m1,m2,chi1,chi2,alpha_GB)[0]
beta = GB_to_ppE(m1,m2,chi1,chi2,alpha_GB)[1]

# Chern-Simons Theory
#b = dCS_to_ppE(m1,m2,chi1,chi2,alpha_CS)[0]
#beta = dCS_to_ppE(m1,m2,chi1,chi2,alpha_CS)[1]

delta_f = sp.delta_f
delta_t = 1.0/4096.0
tp = IFFT_to_TD(m1,m2,b,beta,freq_sp,freq_sc,sp_array,sc_array,delta_t,delta_f)[0]
tc = IFFT_to_TD(m1,m2,b,beta,freq_sp,freq_sc,sp_array,sc_array,delta_t,delta_f)[1]

flow = 30.0
noise = Noise(flow,delta_f,delta_t)

Plus_Signal_in_Noise = noise + tp
Cross_Signal_in_Noise = noise + tc


