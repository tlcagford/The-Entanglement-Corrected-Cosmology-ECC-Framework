import numpy as np
from .entanglement_density_models import EntanglementDensityBase

class OptimizedEarlyDarkEnergyModel(EntanglementDensityBase):
    """
    Enhanced early dark energy-like model with optimized parameters
    for Hubble Tension resolution
    """
    
    def __init__(self, coupling_strength=5.12e-10, peak_redshift=850, 
                 peak_fraction=0.12, decay_width=180):
        super().__init__(coupling_strength, 100)
        self.z_peak = peak_redshift
        self.a_peak = 1/(1 + self.z_peak)
        self.width = decay_width
        self.f_peak = peak_fraction
        
        # Optimization: Stronger peak, earlier transition
        self.enhancement_factor = 2.5  # Boost entanglement effects
        
    def density(self, a):
        z = 1/a - 1
        
        if a <= self.a_peak:
            # Enhanced early universe entanglement
            coherence = np.exp(-((z - self.z_peak)/(2*self.width))**2)
            rho_peak = self.f_peak * self.rho_crit * (self.a_peak/a)**3
            return rho_peak * coherence * self.enhancement_factor
        else:
            # Faster decay to avoid late-time conflicts
            decay_exp = np.exp(-(a - self.a_peak)/0.05)  # Faster decay
            return self.f_peak * self.rho_crit * (self.a_peak/a)**4 * decay_exp

class OptimizedPersistentEntanglementModel(EntanglementDensityBase):
    """
    Enhanced persistent entanglement with optimized evolution
    """
    
    def __init__(self, coupling_strength=8.91e-10, primordial_strength=3.16e-8, 
                 decay_index=0.35):
        super().__init__(coupling_strength, 120)
        self.rho_primordial = primordial_strength * self.rho_crit
        self.n = decay_index
        
        # Optimization: More persistent component
        self.persistence_boost = 1.8
        
    def density(self, a):
        # Enhanced persistent component
        persistent = 0.15 * self.rho_primordial * a**(-3*(1 + self.n)) * self.persistence_boost
        # Modified decaying component
        decaying = 0.85 * self.rho_primordial * np.exp(-8*(1-a)) * a**(-3.2)
        return persistent + decaying

class OptimizedQuantumCoherenceModel(EntanglementDensityBase):
    """
    Enhanced quantum coherence model with optimized parameters
    """
    
    def __init__(self, coupling_strength=2.75e-9, coherence_length=2800, 
                 quantum_fluctuations=1.78e-6):
        super().__init__(coupling_strength, 150)
        self.L_coh = coherence_length
        self.delta_quantum = quantum_fluctuations
        
        # Optimization: Stronger coherence effects
        self.coherence_boost = 3.2
        
    def density(self, a):
        H0 = 2.2e-18
        H_a = H0 * np.sqrt(0.3/a**3 + 0.7)
        
        # Enhanced coherence volume effect
        coherence_volume = (self.L_coh * H_a / 3e5)**3
        rho_quantum = self.delta_quantum * self.rho_crit * coherence_volume
        
        # Modified scaling for better tension resolution
        return rho_quantum * a**(-3.8) * self.coherence_boost

def get_optimized_model(model_name, **kwargs):
    """Factory function for optimized models"""
    models = {
        'early_dark_energy_like': OptimizedEarlyDarkEnergyModel,
        'persistent_entanglement': OptimizedPersistentEntanglementModel,
        'quantum_coherence': OptimizedQuantumCoherenceModel
    }
    
    if model_name not in models:
        raise ValueError(f"Optimized model {model_name} not found")
    
    return models[model_name](**kwargs)
