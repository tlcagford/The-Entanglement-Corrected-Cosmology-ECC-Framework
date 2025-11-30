import numpy as np
from scipy.integrate import solve_ivp
from .entanglement_density_models import get_entanglement_model

class ModifiedFriedmannSolver:
    """
    Solves the Friedmann equations with entanglement density correction
    """
    
    def __init__(self):
        self.H0 = 67.4  # Planck base value (km/s/Mpc)
        self.omega_m = 0.315  # Matter density parameter
        self.omega_lambda = 0.685  # Dark energy density parameter
        self.omega_r = 9.2e-5  # Radiation density parameter
        
    def friedmann_equation(self, a, y, entanglement_model):
        """
        Modified Friedmann equation: H²/H₀² = Ω_m/a³ + Ω_r/a⁴ + Ω_Λ + Ω_ent(a)
        """
        H = y[0]
        
        # Standard components
        standard_terms = (self.omega_m / a**3 + 
                         self.omega_r / a**4 + 
                         self.omega_lambda)
        
        # Entanglement correction from your model
        omega_ent = entanglement_model.density_parameter(a)
        
        # Modified Friedmann equation
        H_squared = standard_terms + omega_ent
        
        return [H * np.sqrt(H_squared)]
    
    def solve_with_entanglement(self, cmb_data, model_name='early_dark_energy_like', **model_kwargs):
        """
        Solve cosmic expansion history with entanglement correction
        """
        # Initialize your entanglement model
        ent_model = get_entanglement_model(model_name, **model_kwargs)
        
        # Scale factor range (from CMB to today)
        a_span = (1/1100, 1.0)  # Recombination to today
        a_eval = np.logspace(np.log10(a_span[0]), 0, 1000)
        
        # Initial condition: H(a_CMB) from standard cosmology
        H_init = [self.H0 * np.sqrt(self.omega_m / a_span[0]**3 + 
                                   self.omega_r / a_span[0]**4 + 
                                   self.omega_lambda)]
        
        # Solve the differential equation
        solution = solve_ivp(
            self.friedmann_equation, 
            a_span, 
            H_init,
            args=(ent_model,),
            t_eval=a_eval,
            method='RK45'
        )
        
        # Extract Hubble parameter today (a=1)
        H0_corrected = np.interp(1.0, solution.t, solution.y[0])
        
        # Calculate sound horizon at recombination (key CMB quantity)
        sound_horizon = self.calculate_sound_horizon(solution, ent_model)
        
        return {
            'H0': H0_corrected,
            'sound_horizon': sound_horizon,
            'scale_factors': solution.t,
            'hubble_parameters': solution.y[0],
            'entanglement_density': [ent_model.density(a) for a in solution.t],
            'model_used': model_name
        }
    
    def calculate_sound_horizon(self, solution, ent_model):
        """Calculate sound horizon at recombination with entanglement corrections"""
        # This would integrate the sound speed with the expansion history
        # Simplified implementation for now
        a_recomb = 1/1100
        H_recomb = np.interp(a_recomb, solution.t, solution.y[0])
        return 144.0 * (67.4 / H_recomb)  # Approximate scaling
