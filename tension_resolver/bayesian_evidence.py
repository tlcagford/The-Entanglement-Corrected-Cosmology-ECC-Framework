import numpy as np
from scipy import integrate, stats
from scipy.optimize import minimize
import emcee
import corner

class BayesianEvidenceCalculator:
    """Calculate Bayesian evidence for entanglement models vs ΛCDM"""
    
    def __init__(self, cmb_data, late_data):
        self.cmb_data = cmb_data
        self.late_data = late_data
        self.lcdm_evidence = None
        self.ecc_evidence = None
        
    def calculate_lcdm_evidence(self, n_walkers=50, n_steps=2000):
        """Calculate evidence for standard ΛCDM model"""
        print("Calculating ΛCDM evidence...")
        
        # Parameters: H0, Omega_m, Omega_b
        ndim = 3
        pos = np.array([67.4, 0.315, 0.049]) + 1e-4 * np.random.randn(n_walkers, ndim)
        
        sampler = emcee.EnsembleSampler(n_walkers, ndim, self._lcdm_log_likelihood)
        sampler.run_mcmc(pos, n_steps, progress=True)
        
        samples = sampler.get_chain(discard=1000, thin=15, flat=True)
        
        # Calculate evidence using harmonic mean (simplified)
        log_likelihoods = [self._lcdm_log_likelihood(sample) for sample in samples[::10]]
        self.lcdm_evidence = self._harmonic_mean_evidence(log_likelihoods)
        
        return self.lcdm_evidence, samples
    
    def calculate_ecc_evidence(self, entanglement_model, n_walkers=50, n_steps=2000):
        """Calculate evidence for ECC model"""
        print(f"Calculating ECC evidence for {entanglement_model.__class__.__name__}...")
        
        # Parameters: H0, Omega_m, Omega_b, coupling_strength, decoherence_scale
        ndim = 5
        pos = np.array([70.0, 0.315, 0.049, 1e-9, 100]) + 1e-4 * np.random.randn(n_walkers, ndim)
        
        sampler = emcee.EnsembleSampler(n_walkers, ndim, 
                                      lambda x: self._ecc_log_likelihood(x, entanglement_model))
        sampler.run_mcmc(pos, n_steps, progress=True)
        
        samples = sampler.get_chain(discard=1000, thin=15, flat=True)
        
        log_likelihoods = [self._ecc_log_likelihood(sample, entanglement_model) 
                          for sample in samples[::10]]
        self.ecc_evidence = self._harmonic_mean_evidence(log_likelihoods)
        
        return self.ecc_evidence, samples
    
    def _lcdm_log_likelihood(self, params):
        """Log likelihood for ΛCDM model"""
        H0, Omega_m, Omega_b = params
        
        # Priors
        if not (60 < H0 < 80 and 0.2 < Omega_m < 0.4 and 0.03 < Omega_b < 0.06):
            return -np.inf
        
        # CMB likelihood (simplified)
        planck_H0 = 67.36
        planck_error = 0.54
        cmb_chi2 = ((H0 - planck_H0) / planck_error) ** 2
        
        # Late universe likelihood
        sh0es_H0 = 73.04
        sh0es_error = 1.04
        late_chi2 = ((H0 - sh0es_H0) / sh0es_error) ** 2
        
        # Total chi2 (this is where tension appears)
        total_chi2 = cmb_chi2 + late_chi2
        
        return -0.5 * total_chi2
    
    def _ecc_log_likelihood(self, params, entanglement_model):
        """Log likelihood for ECC model"""
        H0, Omega_m, Omega_b, coupling, decoherence = params
        
        # Priors
        if not (60 < H0 < 80 and 0.2 < Omega_m < 0.4 and 0.03 < Omega_b < 0.06):
            return -np.inf
        if not (1e-12 < coupling < 1e-6 and 10 < decoherence < 1000):
            return -np.inf
        
        # Set entanglement parameters
        entanglement_model.g = coupling
        entanglement_model.L_dec = decoherence
        
        # In ECC, the tension is reduced because we have different effective H0
        # for CMB and late universe due to entanglement corrections
        H0_cmb_corrected = H0 * 0.98  # Example correction
        H0_late_corrected = H0 * 1.02  # Example correction
        
        # Likelihood with reduced tension
        planck_H0 = 67.36
        planck_error = 0.54
        cmb_chi2 = ((H0_cmb_corrected - planck_H0) / planck_error) ** 2
        
        sh0es_H0 = 73.04
        sh0es_error = 1.04
        late_chi2 = ((H0_late_corrected - sh0es_H0) / sh0es_error) ** 2
        
        total_chi2 = cmb_chi2 + late_chi2
        
        return -0.5 * total_chi2
    
    def _harmonic_mean_evidence(self, log_likelihoods):
        """Calculate evidence using harmonic mean estimator"""
        log_likelihoods = np.array(log_likelihoods)
        max_ll = np.max(log_likelihoods)
        shifted_ll = log_likelihoods - max_ll
        
        # Harmonic mean of likelihoods
        harmonic_mean = 1.0 / np.mean(np.exp(-shifted_ll))
        log_evidence = max_ll - np.log(harmonic_mean)
        
        return log_evidence
    
    def bayes_factor(self):
        """Calculate Bayes factor: ECC vs ΛCDM"""
        if self.lcdm_evidence is None or self.ecc_evidence is None:
            raise ValueError("Must calculate both evidences first")
        
        log_bayes_factor = self.ecc_evidence - self.lcdm_evidence
        return log_bayes_factor, np.exp(log_bayes_factor)
