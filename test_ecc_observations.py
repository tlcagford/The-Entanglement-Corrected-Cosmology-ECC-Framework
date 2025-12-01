"""
Test ECC against observational data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

class ECCObservationalTester:
    def __init__(self):
        # Load or generate standard observational data
        self.setup_data()
    
    def setup_data(self):
        """Setup test observational data"""
        # Example: Type Ia Supernova data (mock)
        z = np.linspace(0.01, 1.5, 30)
        # ΛCDM distance modulus
        mu_lcdm = self.lcdm_distance_modulus(z, Om0=0.3, Ode0=0.7, H0=70)
        # Add some noise
        np.random.seed(42)
        self.sn_data = pd.DataFrame({
            'z': z,
            'mu': mu_lcdm + np.random.normal(0, 0.1, len(z)),
            'mu_err': np.full(len(z), 0.15)
        })
    
    def lcdm_distance_modulus(self, z, Om0, Ode0, H0):
        """ΛCDM distance modulus"""
        from scipy.integrate import quad
        
        def e_inv(z):
            return 1 / np.sqrt(Om0 * (1 + z)**3 + Ode0)
        
        mu = []
        for zi in z:
            dL = (1 + zi) * quad(e_inv, 0, zi)[0]
            mu.append(5 * np.log10(dL * 1e5 * H0 / 70))  # Normalized to H0=70
        return np.array(mu)
    
    def test_supernova_fit(self, ecc_distance_modulus):
        """Test fit to supernova data"""
        print("\nTesting Supernova Data Fit")
        print("-" * 40)
        
        def chi2(params):
            """Calculate chi-squared"""
            mu_pred = ecc_distance_modulus(self.sn_data['z'], *params)
            residuals = self.sn_data['mu'] - mu_pred
            return np.sum((residuals / self.sn_data['mu_err'])**2)
        
        # Initial guess for ECC parameters
        # [H0, Om0, alpha_ent, ...]
        params0 = [70, 0.3, 0.01]
        
        try:
            # Fit using scipy
            popt, pcov = curve_fit(
                ecc_distance_modulus,
                self.sn_data['z'],
                self.sn_data['mu'],
                p0=params0,
                sigma=self.sn_data['mu_err'],
                maxfev=5000
            )
            
            chi2_min = chi2(popt)
            dof = len(self.sn_data) - len(popt)
            reduced_chi2 = chi2_min / dof
            
            print(f"Best-fit parameters: {popt}")
            print(f"χ²/dof = {chi2_min:.2f}/{dof} = {reduced_chi2:.2f}")
            
            if 0.5 < reduced_chi2 < 1.5:
                print("✓ Good fit to SN data (χ²/dof ≈ 1)")
                return True, popt
            else:
                print(f"✗ Poor fit to SN data")
                return False, popt
                
        except Exception as e:
            print(f"✗ SN fit failed: {e}")
            return False, None
    
    def test_cmb_shift(self, ecc_cmb_shift_function):
        """Test CMB shift parameter"""
        print("\nTesting CMB Shift Parameter")
        print("-" * 40)
        
        # Planck 2018: R = 1.7502 ± 0.0046
        planck_R = 1.7502
        planck_R_err = 0.0046
        
        try:
            # Calculate R in ECC for z=1089 (recombination)
            R_ecc = ecc_cmb_shift_function(z=1089)
            
            print(f"ECC R = {R_ecc:.4f}")
            print(f"Planck R = {planck_R:.4f} ± {planck_R_err:.4f}")
            
            sigma_diff = abs(R_ecc - planck_R) / planck_R_err
            
            if sigma_diff < 2:
                print(f"✓ Consistent with Planck ({sigma_diff:.1f}σ)")
                return True
            else:
                print(f"✗ Inconsistent with Planck ({sigma_diff:.1f}σ)")
                return False
                
        except Exception as e:
            print(f"✗ CMB test failed: {e}")
            return False
    
    def test_baryon_acoustic_oscillations(self, ecc_baofunction):
        """Test BAO scale"""
        print("\nTesting BAO Measurements")
        print("-" * 40)
        
        # Example BAO measurements (simplified)
        bao_data = [
            {'z': 0.38, 'DV_rd': 10.23, 'err': 0.17},  # BOSS DR12
            {'z': 0.51, 'DV_rd': 13.36, 'err': 0.21},
            {'z': 0.61, 'DV_rd': 15.45, 'err': 0.22},
        ]
        
        try:
            chi2_total = 0
            for point in bao_data:
                DV_rd_ecc = ecc_baofunction(point['z'])
                residual = DV_rd_ecc - point['DV_rd']
                chi2_total += (residual / point['err'])**2
            
            print(f"BAO χ² = {chi2_total:.2f} for {len(bao_data)} points")
            
            if chi2_total < 10:  # Rough threshold
                print("✓ Consistent with BAO data")
                return True
            else:
                print("✗ Inconsistent with BAO data")
                return False
                
        except Exception as e:
            print(f"✗ BAO test failed: {e}")
            return False

def run_observational_tests(ecc_framework):
    """Run all observational tests"""
    tester = ECCObservationalTester()
    
    print("=" * 60)
    print("ECC Framework - Observational Tests")
    print("=" * 60)
    
    try:
        # Get ECC functions
        mu_ecc = ecc_framework.distance_modulus
        R_ecc = ecc_framework.cmb_shift
        bao_ecc = ecc_framework.bao_scale
        
        # Run tests
        sn_result, best_params = tester.test_supernova_fit(mu_ecc)
        cmb_result = tester.test_cmb_shift(R_ecc)
        bao_result = tester.test_baryon_acoustic_oscillations(bao_ecc)
        
        results = [sn_result, cmb_result, bao_result]
        passed = sum(results)
        
        print("\n" + "=" * 60)
        print(f"Observational Tests: {passed}/{len(results)} passed")
        if best_params is not None:
            print(f"Best-fit parameters: {best_params}")
        print("=" * 60)
        
        return all(results)
        
    except AttributeError as e:
        print(f"✗ Missing ECC functions: {e}")
        return False

if __name__ == "__main__":
    print("Import ECC framework and run tests")
