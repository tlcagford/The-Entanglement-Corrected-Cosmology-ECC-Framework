"""
Modified Friedmann Solver for Entanglement-Corrected Cosmology (ECC) Framework

This module implements the modified Friedmann equations with quantum entanglement
corrections. It provides numerical solvers and analytical approximations for
cosmological evolution with various entanglement models.

Author: T.L. Cagford
Repository: https://github.com/tlcagford/The-Entanglement-Corrected-Cosmology-ECC-Framework
"""

import numpy as np
from typing import Union, Callable, Dict, List, Optional, Tuple
import warnings

class ModifiedFriedmannSolver:
    """
    A solver for the modified Friedmann equation incorporating quantum 
    entanglement/quantum correlation terms in cosmology.
    
    This class computes cosmological evolution including additional
    density contributions from quantum entanglement effects, which may
    act as an effective dark energy component with non-trivial evolution.
    
    Parameters
    ----------
    H0 : float, optional
        Hubble constant at present time (km/s/Mpc), default=70
    Omega_m0 : float, optional
        Present matter density parameter, default=0.3
    Omega_r0 : float, optional
        Present radiation density parameter, default=8.24e-5
    Omega_Lambda0 : float, optional
        Present dark energy density parameter, default=0.7
    Omega_k0 : float, optional
        Present curvature density parameter, default=0
    
    Attributes
    ----------
    H0 : float
        Hubble constant (km/s/Mpc)
    Omega_m0 : float
        Present matter density parameter
    Omega_r0 : float
        Present radiation density parameter
    Omega_Lambda0 : float
        Present cosmological constant density parameter
    Omega_k0 : float
        Present curvature density parameter
    Omega_total0 : float
        Sum of all density parameters at present
        
    Notes
    -----
    The modified Friedmann equation implemented is:
    
    H(a)^2 = H0^2 * [Ω_m0 * a^{-3} + Ω_r0 * a^{-4} + Ω_k0 * a^{-2} + Ω_Λ0 + Ω_ent(a)]
    
    where Ω_ent(a) represents the entanglement correction term.
    """
    
    def __init__(self, 
                 H0: float = 70.0, 
                 Omega_m0: float = 0.3,
                 Omega_r0: float = 8.24e-5,
                 Omega_Lambda0: float = 0.7,
                 Omega_k0: float = 0.0):
        """Initialize cosmological parameters."""
        self.H0 = H0
        self.Omega_m0 = Omega_m0
        self.Omega_r0 = Omega_r0
        self.Omega_Lambda0 = Omega_Lambda0
        self.Omega_k0 = Omega_k0
        
        # Calculate total density parameter (should be 1 for flat universe)
        self.Omega_total0 = Omega_m0 + Omega_r0 + Omega_Lambda0 + Omega_k0
        
        # Check for flatness (within tolerance)
        if abs(self.Omega_total0 - 1.0) > 1e-3:
            warnings.warn(f"Total density parameter Ω_total0 = {self.Omega_total0:.4f} "
                         f"deviates significantly from 1 (flat universe)")
    
    def standard_density_terms(self, a: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute standard ΛCDM density terms as functions of scale factor.
        
        Parameters
        ----------
        a : float or np.ndarray
            Scale factor (normalized to a=1 today)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'matter': Ω_m0 * a^{-3}
            - 'radiation': Ω_r0 * a^{-4}
            - 'curvature': Ω_k0 * a^{-2}
            - 'cosmological_constant': Ω_Λ0 (constant)
        """
        return {
            'matter': self.Omega_m0 / a**3,
            'radiation': self.Omega_r0 / a**4,
            'curvature': self.Omega_k0 / a**2,
            'cosmological_constant': self.Omega_Lambda0 * np.ones_like(a)
        }
    
    def entanglement_density(self, 
                            a: Union[float, np.ndarray], 
                            model_name: str = 'early_dark_energy_like',
                            **kwargs) -> Union[float, np.ndarray]:
        """
        Compute entanglement density term Ω_ent(a) for various models.
        
        Parameters
        ----------
        a : float or np.ndarray
            Scale factor (normalized to a=1 today)
        model_name : str, optional
            Name of entanglement model. Options:
            - 'early_dark_energy_like': Gaussian peak at early times
            - 'persistent_entanglement': Power-law decaying entanglement
            - 'quantum_coherence': Oscillatory quantum coherence effects
            - 'step_function': Sudden phase transition model
            - 'logarithmic': Logarithmic dependence on scale factor
            - 'custom': User-defined model (requires custom_function)
        **kwargs
            Model-specific parameters:
            - amplitude (float): Amplitude of entanglement effect
            - peak_scale (float): Scale factor at peak (for Gaussian)
            - width (float): Width of Gaussian peak
            - decay_exponent (float): Decay exponent for power-law
            - frequency (float): Frequency for oscillatory models
            - transition_scale (float): Transition scale for step function
            - custom_function (callable): Custom Ω_ent(a) function
            
        Returns
        -------
        float or np.ndarray
            Entanglement density contribution Ω_ent(a)
        
        Raises
        ------
        ValueError
            If model_name is not recognized
            
        Notes
        -----
        The entanglement density is dimensionless and adds to the total
        energy density in the Friedmann equation.
        """
        # Default parameters
        amplitude = kwargs.get('amplitude', 0.01)
        peak_scale = kwargs.get('peak_scale', 1e-3)  # a ~ 0.001 (z ~ 1000)
        width = kwargs.get('width', 0.5)
        decay_exponent = kwargs.get('decay_exponent', 1.0)
        frequency = kwargs.get('frequency', 10.0)
        transition_scale = kwargs.get('transition_scale', 0.1)
        
        if model_name == 'early_dark_energy_like':
            # Gaussian peak centered at early times
            return amplitude * np.exp(-((np.log(a) - np.log(peak_scale))**2) / (2 * width**2))
            
        elif model_name == 'persistent_entanglement':
            # Power-law decaying entanglement
            return amplitude / (a**decay_exponent)
            
        elif model_name == 'quantum_coherence':
            # Oscillatory behavior from quantum coherence
            return amplitude * (1 + np.sin(frequency * np.log(a)))
            
        elif model_name == 'step_function':
            # Sudden phase transition model
            return amplitude * (0.5 * (1 + np.tanh((np.log(a) - np.log(transition_scale)) / 0.1)))
            
        elif model_name == 'logarithmic':
            # Logarithmic dependence (motivated by entanglement entropy)
            return amplitude * (1 - np.log(a))
            
        elif model_name == 'custom':
            custom_func = kwargs.get('custom_function')
            if custom_func is None:
                raise ValueError("custom_function must be provided for 'custom' model")
            return custom_func(a)
            
        else:
            raise ValueError(f"Unknown model_name: {model_name}. "
                           f"Available models: {self.list_available_models()}")
    
    def H(self, 
          a: Union[float, np.ndarray], 
          model_name: str = 'early_dark_energy_like',
          **kwargs) -> Union[float, np.ndarray]:
        """
        Compute Hubble parameter H(a) including entanglement effects.
        
        Parameters
        ----------
        a : float or np.ndarray
            Scale factor (normalized to a=1 today)
        model_name : str, optional
            Name of entanglement model
        **kwargs
            Parameters passed to entanglement_density method
        
        Returns
        -------
        float or np.ndarray
            Hubble parameter H(a) in km/s/Mpc
            
        Examples
        --------
        >>> solver = ModifiedFriedmannSolver()
        >>> H = solver.H(0.5, model_name='early_dark_energy_like')
        >>> H_values = solver.H(np.linspace(0.1, 1, 100), model_name='quantum_coherence')
        """
        # Compute standard density terms
        densities = self.standard_density_terms(a)
        
        # Compute entanglement density
        omega_ent = self.entanglement_density(a, model_name, **kwargs)
        
        # Sum all contributions
        total_density = (densities['matter'] + 
                        densities['radiation'] + 
                        densities['curvature'] + 
                        densities['cosmological_constant'] + 
                        omega_ent)
        
        # Ensure non-negative
        total_density = np.maximum(total_density, 0)
        
        return self.H0 * np.sqrt(total_density)
    
    def H_z(self, 
           z: Union[float, np.ndarray], 
           model_name: str = 'early_dark_energy_like',
           **kwargs) -> Union[float, np.ndarray]:
        """
        Compute Hubble parameter H(z) as function of redshift.
        
        Parameters
        ----------
        z : float or np.ndarray
            Redshift
        model_name : str, optional
            Name of entanglement model
        **kwargs
            Parameters passed to entanglement_density method
        
        Returns
        -------
        float or np.ndarray
            Hubble parameter H(z) in km/s/Mpc
        """
        a = 1.0 / (1.0 + z)
        return self.H(a, model_name, **kwargs)
    
    def critical_density(self, 
                        a: Union[float, np.ndarray],
                        model_name: str = 'early_dark_energy_like',
                        **kwargs) -> Union[float, np.ndarray]:
        """
        Compute critical density ρ_crit(a) including entanglement effects.
        
        Parameters
        ----------
        a : float or np.ndarray
            Scale factor
        model_name : str, optional
            Name of entanglement model
        **kwargs
            Parameters passed to entanglement_density method
        
        Returns
        -------
        float or np.ndarray
            Critical density in units where H0^2 sets the scale
        """
        H_a = self.H(a, model_name, **kwargs)
        # ρ_crit = 3H^2/(8πG), but we return in dimensionless units
        # relative to standard critical density today
        return (H_a / self.H0)**2
    
    def omega_components(self, 
                        a: Union[float, np.ndarray],
                        model_name: str = 'early_dark_energy_like',
                        **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute all density parameters as functions of scale factor.
        
        Parameters
        ----------
        a : float or np.ndarray
            Scale factor
        model_name : str, optional
            Name of entanglement model
        **kwargs
            Parameters passed to entanglement_density method
        
        Returns
        -------
        dict
            Dictionary containing Ω_i(a) for all components
        """
        # Compute Hubble parameter
        H_a = self.H(a, model_name, **kwargs)
        
        # Compute density parameters
        densities = self.standard_density_terms(a)
        omega_ent = self.entanglement_density(a, model_name, **kwargs)
        
        # Normalize by critical density
        rho_crit = self.critical_density(a, model_name, **kwargs)
        
        components = {}
        for name, value in densities.items():
            components[name] = value / rho_crit
        components['entanglement'] = omega_ent / rho_crit
        
        return components
    
    def solve_cosmological_evolution(self,
                                    a_range: Tuple[float, float] = (1e-10, 1.0),
                                    n_points: int = 1000,
                                    model_name: str = 'early_dark_energy_like',
                                    **kwargs) -> Dict[str, np.ndarray]:
        """
        Solve for cosmological evolution over a range of scale factors.
        
        Parameters
        ----------
        a_range : tuple, optional
            (a_min, a_max) range for solution
        n_points : int, optional
            Number of points in solution
        model_name : str, optional
            Name of entanglement model
        **kwargs
            Parameters passed to entanglement_density method
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'a': Scale factors
            - 'z': Redshifts
            - 'H': Hubble parameters
            - 'Omega_i': Density parameters for all components
        """
        a_min, a_max = a_range
        
        # Create scale factor array (logarithmic spacing for early times)
        if a_min < 0.01:
            a = np.logspace(np.log10(a_min), np.log10(a_max), n_points)
        else:
            a = np.linspace(a_min, a_max, n_points)
        
        # Compute Hubble parameter
        H = self.H(a, model_name, **kwargs)
        
        # Compute density parameters
        omega_dict = self.omega_components(a, model_name, **kwargs)
        
        # Compute redshift
        z = 1.0 / a - 1.0
        
        # Package results
        results = {
            'scale_factor': a,
            'redshift': z,
            'Hubble': H,
            'density_parameters': omega_dict,
            'entanglement_model': model_name,
            'cosmological_parameters': {
                'H0': self.H0,
                'Omega_m0': self.Omega_m0,
                'Omega_r0': self.Omega_r0,
                'Omega_Lambda0': self.Omega_Lambda0,
                'Omega_k0': self.Omega_k0
            }
        }
        
        return results
    
    def compare_with_lcdm(self,
                         a: Union[float, np.ndarray],
                         model_name: str = 'early_dark_energy_like',
                         **kwargs) -> Dict[str, np.ndarray]:
        """
        Compare ECC model with standard ΛCDM predictions.
        
        Parameters
        ----------
        a : float or np.ndarray
            Scale factor
        model_name : str, optional
            Name of entanglement model
        **kwargs
            Parameters passed to entanglement_density method
        
        Returns
        -------
        dict
            Dictionary containing comparison metrics:
            - 'H_ECC': Hubble parameter in ECC model
            - 'H_LCDM': Hubble parameter in ΛCDM
            - 'relative_difference': (H_ECC - H_LCDM) / H_LCDM
            - 'Omega_ent': Entanglement density parameter
        """
        # ECC Hubble parameter
        H_ecc = self.H(a, model_name, **kwargs)
        
        # ΛCDM Hubble parameter (no entanglement)
        H_lcdm = self.H0 * np.sqrt(
            self.Omega_m0 / a**3 +
            self.Omega_r0 / a**4 +
            self.Omega_k0 / a**2 +
            self.Omega_Lambda0
        )
        
        # Entanglement density
        omega_ent = self.entanglement_density(a, model_name, **kwargs)
        
        return {
            'H_ECC': H_ecc,
            'H_LCDM': H_lcdm,
            'relative_difference': (H_ecc - H_lcdm) / H_lcdm,
            'Omega_ent': omega_ent,
            'scale_factor': a
        }
    
    def compute_luminosity_distance(self,
                                   z: Union[float, np.ndarray],
                                   model_name: str = 'early_dark_energy_like',
                                   n_points: int = 1000,
                                   **kwargs) -> Union[float, np.ndarray]:
        """
        Compute luminosity distance D_L(z) in ECC cosmology.
        
        Parameters
        ----------
        z : float or np.ndarray
            Redshift(s) at which to compute distance
        model_name : str, optional
            Name of entanglement model
        n_points : int, optional
            Number of integration points for numerical integration
        **kwargs
            Parameters passed to entanglement_density method
        
        Returns
        -------
        float or np.ndarray
            Luminosity distance in Mpc
        """
        # Convert to array for consistent processing
        z = np.atleast_1d(z)
        
        # Speed of light in km/s
        c = 299792.458
        
        # Initialize results array
        d_L = np.zeros_like(z)
        
        # For each redshift, integrate 1/H(z')
        for i, zi in enumerate(z):
            if zi == 0:
                d_L[i] = 0
                continue
                
            # Create integration grid
            z_grid = np.linspace(0, zi, n_points)
            a_grid = 1.0 / (1.0 + z_grid)
            
            # Compute Hubble parameter on grid
            H_grid = self.H(a_grid, model_name, **kwargs)
            
            # Integrate using trapezoidal rule
            integrand = 1.0 / H_grid
            integral = np.trapz(integrand, z_grid)
            
            # Luminosity distance
            d_L[i] = c * (1 + zi) * integral
        
        return d_L[0] if d_L.size == 1 else d_L
    
    @staticmethod
    def list_available_models() -> List[str]:
        """List all available entanglement models."""
        return [
            'early_dark_energy_like',
            'persistent_entanglement',
            'quantum_coherence',
            'step_function',
            'logarithmic',
            'custom'
        ]
    
    def validate_model_parameters(self, 
                                 model_name: str,
                                 **kwargs) -> bool:
        """
        Validate parameters for a given entanglement model.
        
        Parameters
        ----------
        model_name : str
            Name of entanglement model
        **kwargs
            Parameters to validate
        
        Returns
        -------
        bool
            True if parameters are valid
            
        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if model_name == 'custom':
            if 'custom_function' not in kwargs:
                raise ValueError("custom_function must be provided for 'custom' model")
            if not callable(kwargs['custom_function']):
                raise ValueError("custom_function must be callable")
        
        # Check amplitude is positive
        amplitude = kwargs.get('amplitude', 0.01)
        if amplitude < 0:
            raise ValueError("amplitude must be non-negative")
        
        # Check scale factors are positive
        for key in ['peak_scale', 'transition_scale']:
            if key in kwargs and kwargs[key] <= 0:
                raise ValueError(f"{key} must be positive")
        
        return True
    
    def to_dict(self) -> Dict:
        """Export cosmological parameters to dictionary."""
        return {
            'H0': self.H0,
            'Omega_m0': self.Omega_m0,
            'Omega_r0': self.Omega_r0,
            'Omega_Lambda0': self.Omega_Lambda0,
            'Omega_k0': self.Omega_k0,
            'Omega_total0': self.Omega_total0
        }
    
    @classmethod
    def from_dict(cls, params: Dict) -> 'ModifiedFriedmannSolver':
        """Create solver from dictionary of parameters."""
        return cls(
            H0=params.get('H0', 70.0),
            Omega_m0=params.get('Omega_m0', 0.3),
            Omega_r0=params.get('Omega_r0', 8.24e-5),
            Omega_Lambda0=params.get('Omega_Lambda0', 0.7),
            Omega_k0=params.get('Omega_k0', 0.0)
        )


# Convenience function for quick calculations
def compute_H_with_entanglement(a: Union[float, np.ndarray],
                               model_name: str = 'early_dark_energy_like',
                               H0: float = 70.0,
                               Omega_m0: float = 0.3,
                               Omega_Lambda0: float = 0.7,
                               **kwargs) -> Union[float, np.ndarray]:
    """
    Quick computation of H(a) with entanglement corrections.
    
    Parameters
    ----------
    a : float or np.ndarray
        Scale factor
    model_name : str, optional
        Entanglement model name
    H0 : float, optional
        Hubble constant
    Omega_m0 : float, optional
        Matter density parameter
    Omega_Lambda0 : float, optional
        Cosmological constant density parameter
    **kwargs
        Additional parameters for entanglement model
    
    Returns
    -------
    float or np.ndarray
        Hubble parameter H(a)
    """
    solver = ModifiedFriedmannSolver(H0=H0, 
                                    Omega_m0=Omega_m0, 
                                    Omega_Lambda0=Omega_Lambda0)
    return solver.H(a, model_name, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("Modified Friedmann Solver for ECC Framework")
    print("=" * 50)
    
    # Create solver
    solver = ModifiedFriedmannSolver(H0=70, Omega_m0=0.3, Omega_Lambda0=0.7)
    
    # Test different models
    a_test = np.logspace(-4, 0, 50)  # From early universe to today
    
    print("\nTesting different entanglement models:")
    print("-" * 40)
    
    models = ['early_dark_energy_like', 'persistent_entanglement', 'quantum_coherence']
    
    for model in models:
        H = solver.H(a_test, model_name=model)
        print(f"{model:30s}: H(a=0.1) = {solver.H(0.1, model_name=model):.2f} km/s/Mpc")
    
    # Compare with ΛCDM
    print("\nComparison with ΛCDM at a=0.5:")
    print("-" * 40)
    
    comparison = solver.compare_with_lcdm(0.5, model_name='early_dark_energy_like')
    print(f"H_ECC  = {comparison['H_ECC']:.2f} km/s/Mpc")
    print(f"H_ΛCDM = {comparison['H_LCDM']:.2f} km/s/Mpc")
    print(f"Relative difference: {comparison['relative_difference']*100:.2f}%")
    print(f"Ω_ent(a=0.5) = {comparison['Omega_ent']:.6f}")
