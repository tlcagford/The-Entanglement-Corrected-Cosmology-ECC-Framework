import numpy as np
from scipy import integrate
from abc import ABC, abstractmethod

class EntanglementDensityBase(ABC):
    """Base class for photon-dark photon entanglement density models"""
    
    def __init__(self, coupling_strength=1e-9, decoherence_scale=100):
        self.g = coupling_strength  # Dark photon coupling strength
        self.L_dec = decoherence_scale  # Decoherence scale in Mpc
        self.rho_crit = 8.62e-27  # Critical density today in kg/m³
        
    @abstractmethod
    def density(self, a):
        """Return entanglement energy density as function of scale factor a"""
        pass
    
    def density_parameter(self, a):
        """Return Omega_ent(a) = rho_ent(a) / rho_crit"""
        return self.density(a) / self.rho_crit

class EarlyDarkEnergyLikeModel(EntanglementDensityBase):
    """
    Model where entanglement density peaks in early universe then decays
    Mimics early dark energy behavior but with quantum entanglement origin
    """
    
    def __init__(self, peak_redshift=1100, decay_width=100, peak_fraction=0.05):
        super().__init__()
        self.z_peak = peak_redshift  # Redshift of peak density (near recombination)
        self.a_peak = 1/(1 + self.z_peak)
        self.width = decay_width
        self.f_peak = peak_fraction  # Peak fraction of total energy density
        
    def density(self, a):
        # Gaussian-like peak around recombination era
        z = 1/a - 1
        z_peak = self.z_peak
        
        # Entanglement density peaks during recombination then decays
        if a <= self.a_peak:
            # Before/during peak: quantum coherence builds up
            coherence_factor = np.exp(-((z - z_peak)/(2*self.width))**2)
            rho_peak = self.f_peak * self.rho_crit * (self.a_peak/a)**3
            return rho_peak * coherence_factor
        else:
            # After peak: entanglement decays due to cosmic expansion
            decay_exp = np.exp(-(a - self.a_peak)/0.1)
            return self.f_peak * self.rho_crit * (self.a_peak/a)**4 * decay_exp

class PersistentEntanglementModel(EntanglementDensityBase):
    """
    Model where entanglement density persists and evolves with scale factor
    Represents ongoing photon-dark photon entanglement throughout cosmic history
    """
    
    def __init__(self, primordial_strength=1e-8, decay_index=0.5):
        super().__init__()
        self.rho_primordial = primordial_strength * self.rho_crit
        self.n = decay_index
        
    def density(self, a):
        # Persistent component + decaying component
        persistent = 0.1 * self.rho_primordial * a**(-3*(1 + self.n))
        decaying = 0.9 * self.rho_primordial * np.exp(-10*(1-a)) * a**(-4)
        return persistent + decaying

class QuantumCoherenceModel(EntanglementDensityBase):
    """
    Based on Stellaris QED Engine principles
    Treats entanglement as a fundamental quantum coherence field
    """
    
    def __init__(self, coherence_length=1e3, quantum_fluctuations=1e-5):
        super().__init__()
        self.L_coh = coherence_length  # Coherence length in Mpc
        self.delta_quantum = quantum_fluctuations
        
    def density(self, a):
        # Quantum coherence scales with horizon size
        H0 = 2.2e-18  # Hubble constant in s^-1
        H_a = H0 * np.sqrt(0.3/a**3 + 0.7)  # Hubble parameter at scale factor a
        
        # Coherence volume effect
        coherence_volume = (self.L_coh * H_a / 3e5)**3  # Normalized volume
        
        # Entanglement density from quantum coherence
        rho_quantum = self.delta_quantum * self.rho_crit * coherence_volume
        
        # Scale with expansion
        return rho_quantum * a**(-3.5)  # Intermediate scaling

# Factory function for easy access
def get_entanglement_model(model_name, **kwargs):
    models = {
        'early_dark_energy_like': EarlyDarkEnergyLikeModel,
        'persistent_entanglement': PersistentEntanglementModel, 
        'quantum_coherence': QuantumCoherenceModel
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)
