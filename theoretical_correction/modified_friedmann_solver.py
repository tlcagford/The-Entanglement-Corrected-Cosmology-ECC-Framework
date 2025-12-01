import numpy as np

class ModifiedFriedmannSolver:
    def __init__(self):
        self.H0 = 70  # km/s/Mpc (example value; can be optimized)
        self.Omega_m = 0.3
        self.Omega_r = 0  # Approximate, ignoring radiation for simplicity
        self.Omega_Lambda = 0.7

    def rho_ent(self, a, model_name):
        """Entanglement density term (as Omega_ent(a))"""
        if model_name == 'early_dark_energy_like':
            # Gaussian peak at early times (a ~ 0.001, z ~ 1000)
            return 0.01 * np.exp( - ((np.log(a) + 8)**2) / 0.5 )
        elif model_name == 'persistent_entanglement':
            # Example: Decays slowly
            return 0.005 / a
        elif model_name == 'quantum_coherence':
            # Example: Oscillatory
            return 0.002 * (1 + np.sin(10 * np.log(a)))
        else:
            return 0

    def H(self, a, model_name):
        """Algebraic computation of H(a)"""
        return self.H0 * np.sqrt(
            self.Omega_m / a**3 +
            self.Omega_r / a**4 +
            self.Omega_Lambda +
            self.rho_ent(a, model_name)
        )

    def solve_with_entanglement(self, a_values, model_name, entanglement_model=None):
        """Compute H(a) for given scale factors"""
        if entanglement_model is not None:
            # If custom model provided, use it for rho_ent (assuming it's a function of a)
            def rho_ent_custom(a):
                return entanglement_model(a)
            H_values = [self.H0 * np.sqrt(
                self.Omega_m / a**3 +
                self.Omega_r / a**4 +
                self.Omega_Lambda +
                rho_ent_custom(a)
            ) for a in a_values]
        else:
            H_values = [self.H(a, model_name) for a in a_values]
        return {'H0': self.H0, 'H_values': H_values}
