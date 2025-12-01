"""
Scientific validation tests for ECC Framework
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

class ECCValidator:
    def __init__(self):
        """Initialize with standard cosmological parameters"""
        # Planck 2018 parameters
        self.H0 = 67.4  # km/s/Mpc
        self.h = self.H0 / 100  # dimensionless Hubble parameter
        self.Om0 = 0.315  # Matter density parameter
        self.Ode0 = 0.685  # Dark energy density parameter
        self.Ogamma0 = 5.38e-5  # Photon density (CMB)
        self.Onu0 = 3.65e-5  # Neutrino density
        self.Ob0 = 0.0493  # Baryon density
        
        # Convert to SI units for calculations
        self.H0_si = self.H0 * 1000 / 3.086e22  # s^-1
        self.critical_density = 3 * self.H0_si**2 / (8 * np.pi * 6.67430e-11)
    
    def lcdm_hubble(self, a, Om0, Ode0):
        """Standard ΛCDM Hubble parameter"""
        return self.H0 * np.sqrt(Om0 * a**-3 + Ode0)
    
    def test_lcdm_recovery(self, ecc_hubble_func):
        """
        Test that ECC reduces to ΛCDM for zero entanglement
        Parameters should be provided by ECC framework
        """
        print("\nTesting ΛCDM Limit Recovery")
        print("-" * 40)
        
        a_values = np.logspace(-3, 0, 100)  # Scale factor from 0.001 to 1
        
        # This requires accessing ECC's Hubble function
        # Assuming ECC has parameter alpha_ent for entanglement strength
        try:
            # Test with alpha_ent = 0
            H_ecc_zero = [ecc_hubble_func(a, alpha_ent=0) for a in a_values]
            H_lcdm = self.lcdm_hubble(a_values, self.Om0, self.Ode0)
            
            # Calculate relative difference
            rel_diff = np.abs(H_ecc_zero - H_lcdm) / H_lcdm
            max_diff = np.max(rel_diff)
            
            print(f"Maximum relative difference: {max_diff:.2e}")
            
            if max_diff < 1e-10:
                print("✓ ECC correctly reduces to ΛCDM when α_ent → 0")
                return True
            else:
                print("✗ ECC doesn't properly reduce to ΛCDM")
                return False
                
        except Exception as e:
            print(f"✗ ΛCDM recovery test failed: {e}")
            return False
    
    def test_energy_conservation(self, rho_func, pressure_func):
        """Test energy conservation: ρ' + 3H(ρ + p) = 0"""
        print("\nTesting Energy Conservation")
        print("-" * 40)
        
        a_values = np.logspace(-2, 0, 50)
        
        try:
            conservation_violations = []
            
            for a in a_values:
                H = self.lcdm_hubble(a, self.Om0, self.Ode0)
                rho = rho_func(a)
                p = pressure_func(a)
                
                # Numerical derivative of density
                da = 1e-6
                rho_plus = rho_func(a + da)
                rho_minus = rho_func(a - da)
                drho_da = (rho_plus - rho_minus) / (2 * da)
                
                # Convert to time derivative: dρ/dt = H*a*dρ/da
                dp_dt = H * a * drho_da
                
                # Conservation equation: dρ/dt + 3H(ρ + p) = 0
                conservation_eq = dp_dt + 3 * H * (rho + p)
                
                conservation_violations.append(abs(conservation_eq))
            
            max_violation = np.max(conservation_violations)
            print(f"Maximum conservation violation: {max_violation:.2e}")
            
            if max_violation < 1e-10:
                print("✓ Energy conservation satisfied")
                return True
            else:
                print("✗ Energy conservation violated")
                return False
                
        except Exception as e:
            print(f"✗ Conservation test failed: {e}")
            return False
    
    def test_positive_energy(self, rho_func):
        """Test that energy densities remain positive"""
        print("\nTesting Positive Energy Conditions")
        print("-" * 40)
        
        a_values = np.logspace(-4, 2, 100)  # From early to late universe
        
        try:
            densities = [rho_func(a) for a in a_values]
            min_density = min(densities)
            
            print(f"Minimum density: {min_density:.2e}")
            
            if min_density > 0:
                print("✓ All energy densities positive")
                return True
            else:
                print("✗ Negative energy densities found")
                return False
                
        except Exception as e:
            print(f"✗ Positive energy test failed: {e}")
            return False
    
    def test_causality(self, sound_speed_func):
        """Test that sound speed doesn't exceed speed of light"""
        print("\nTesting Causality (Sound Speed ≤ c)")
        print("-" * 40)
        
        a_values = np.logspace(-2, 0, 50)
        
        try:
            cs_values = [sound_speed_func(a) for a in a_values]
            max_cs = max(cs_values)
            
            print(f"Maximum sound speed: {max_cs:.3f}c")
            
            if max_cs <= 1:
                print("✓ Causality preserved (cs ≤ c)")
                return True
            else:
                print("✗ Causality violated (cs > c)")
                return False
                
        except Exception as e:
            print(f"✗ Causality test failed: {e}")
            return False
    
    def test_age_universe(self, H_func):
        """Calculate age of universe in ECC"""
        print("\nTesting Age of Universe Calculation")
        print("-" * 40)
        
        try:
            # Age from integration: t = ∫ da/(aH(a))
            a_values = np.logspace(-8, 0, 1000)
            H_values = [H_func(a) for a in a_values]
            
            # Integrate using trapezoidal rule
            integrand = 1 / (a_values * H_values)
            age_gyr = np.trapz(integrand, np.log(a_values))  # Integration in log space
            
            # Convert to Gyr (H in km/s/Mpc)
            age_gyr = age_gyr / self.h * 9.778  # Conversion factor
            
            print(f"ECC Universe age: {age_gyr:.2f} Gyr")
            
            # Compare with Planck 2018: 13.787 ± 0.020 Gyr
            planck_age = 13.787
            
            if abs(age_gyr - planck_age) < 0.5:  # Within 0.5 Gyr tolerance
                print("✓ Age consistent with Planck")
                return True
            else:
                print(f"✗ Age differs from Planck ({planck_age} Gyr)")
                return False
                
        except Exception as e:
            print(f"✗ Age calculation failed: {e}")
            return False

def run_scientific_tests(ecc_framework):
    """Run all scientific tests on ECC framework"""
    validator = ECCValidator()
    
    # These functions need to be imported from ECC
    # Assuming ecc_framework provides:
    # - hubble(a, params)
    # - energy_density(a, params)
    # - pressure(a, params)
    # - sound_speed(a, params)
    
    print("=" * 60)
    print("ECC Framework - Scientific Validation")
    print("=" * 60)
    
    # Try to get functions from ECC framework
    try:
        H_ecc = ecc_framework.hubble
        rho_ecc = ecc_framework.energy_density
        p_ecc = ecc_framework.pressure
        cs_ecc = ecc_framework.sound_speed
        
        tests = [
            lambda: validator.test_lcdm_recovery(H_ecc),
            lambda: validator.test_energy_conservation(rho_ecc, p_ecc),
            lambda: validator.test_positive_energy(rho_ecc),
            lambda: validator.test_causality(cs_ecc),
            lambda: validator.test_age_universe(H_ecc),
        ]
        
        results = []
        for i, test in enumerate(tests):
            try:
                results.append(test())
            except Exception as e:
                print(f"✗ Test {i+1} failed: {e}")
                results.append(False)
        
        passed = sum(results)
        total = len(results)
        
        print("\n" + "=" * 60)
        print(f"Scientific Tests: {passed}/{total} passed")
        print("=" * 60)
        
        return all(results)
        
    except AttributeError as e:
        print(f"✗ Could not access ECC functions: {e}")
        print("Please ensure ECC framework provides required functions")
        return False

if __name__ == "__main__":
    # This requires an actual ECC framework import
    print("This test requires importing the ECC framework")
    print("Please modify to import your ECC implementation")
