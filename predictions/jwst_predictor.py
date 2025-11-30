import numpy as np
import matplotlib.pyplot as plt

class JWSTPredictor:
    """Predict JWST observations using ECC framework"""
    
    def __init__(self, friedmann_solver, entanglement_model):
        self.solver = friedmann_solver
        self.entanglement_model = entanglement_model
        
    def predict_high_z_hubble_flow(self, z_range=np.linspace(0.1, 3.0, 50)):
        """Predict Hubble flow measurements at high redshift"""
        print("Predicting JWST Hubble flow measurements...")
        
        results = []
        for z in z_range:
            a = 1 / (1 + z)
            
            # Calculate H(z) with entanglement correction
            hubble_flow = self._calculate_hubble_flow(a)
            
            # Predict observable quantities
            distance_modulus = self._calculate_distance_modulus(z)
            angular_diameter_distance = self._calculate_angular_diameter_distance(z)
            
            results.append({
                'redshift': z,
                'H(z)': hubble_flow,
                'distance_modulus': distance_modulus,
                'angular_diameter_distance': angular_diameter_distance,
                'entanglement_correction': self.entanglement_model.density(a)
            })
        
        return pd.DataFrame(results)
    
    def _calculate_hubble_flow(self, a):
        """Calculate H(z) with entanglement corrections"""
        # Solve modified Friedmann equation at scale factor a
        H = self.solver.H0 * np.sqrt(
            self.solver.omega_m / a**3 +
            self.solver.omega_r / a**4 +
            self.solver.omega_lambda +
            self.entanglement_model.density_parameter(a)
        )
        return H
    
    def _calculate_distance_modulus(self, z):
        """Calculate distance modulus μ = 5log10(d_L/10pc)"""
        d_L = self._luminosity_distance(z)  # Luminosity distance in Mpc
        return 5 * np.log10(d_L * 1e6)  # Convert to parsecs
    
    def _luminosity_distance(self, z):
        """Calculate luminosity distance with entanglement corrections"""
        # Numerical integration of 1/H(z) with entanglement
        a_values = np.linspace(1/(1+z), 1.0, 1000)
        integrand = 1.0 / np.array([self._calculate_hubble_flow(a) for a in a_values])
        
        comoving_distance = np.trapz(integrand, a_values)
        return comoving_distance * (1 + z)
    
    def _calculate_angular_diameter_distance(self, z):
        """Calculate angular diameter distance d_A = d_L/(1+z)²"""
        d_L = self._luminosity_distance(z)
        return d_L / (1 + z)**2
    
    def plot_jwst_predictions(self, z_max=3.0):
        """Generate prediction plots for JWST observations"""
        predictions = self.predict_high_z_hubble_flow(z_range=np.linspace(0.01, z_max, 100))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Hubble flow H(z)
        axes[0,0].plot(predictions['redshift'], predictions['H(z)'], 'b-', linewidth=2, label='ECC Prediction')
        axes[0,0].set_xlabel('Redshift z')
        axes[0,0].set_ylabel('H(z) [km/s/Mpc]')
        axes[0,0].set_title('JWST: Hubble Flow Predictions')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # Distance modulus
        axes[0,1].plot(predictions['redshift'], predictions['distance_modulus'], 'r-', linewidth=2)
        axes[0,1].set_xlabel('Redshift z')
        axes[0,1].set_ylabel('Distance Modulus μ')
        axes[0,1].set_title('Distance-Redshift Relation')
        axes[0,1].grid(True, alpha=0.3)
        
        # Entanglement correction evolution
        axes[1,0].plot(predictions['redshift'], predictions['entanglement_correction'], 'g-', linewidth=2)
        axes[1,0].set_xlabel('Redshift z')
        axes[1,0].set_ylabel('ρ_ent [kg/m³]')
        axes[1,0].set_title('Entanglement Density Evolution')
        axes[1,0].set_yscale('log')
        axes[1,0].grid(True, alpha=0.3)
        
        # Comparison with ΛCDM
        z_range = predictions['redshift']
        H_lcdm = self.solver.H0 * np.sqrt(
            self.solver.omega_m * (1 + z_range)**3 + self.solver.omega_lambda
        )
        
        axes[1,1].plot(z_range, predictions['H(z)'], 'b-', label='ECC')
        axes[1,1].plot(z_range, H_lcdm, 'k--', label='ΛCDM')
        axes[1,1].set_xlabel('Redshift z')
        axes[1,1].set_ylabel('H(z) [km/s/Mpc]')
        axes[1,1].set_title('ECC vs ΛCDM Hubble Flow')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('OUTPUT/jwst_predictions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return predictions
