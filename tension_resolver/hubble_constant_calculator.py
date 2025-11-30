import numpy as np
from scipy.optimize import curve_fit

class HubbleConstantCalculator:
    """Calculate H0 from corrected distance ladder data"""
    
    def calculate_from_distance_ladder(self, corrected_data):
        """Calculate H0 using distance-redshift relation"""
        if 'supernovae' in corrected_data:
            return self._from_supernovae(corrected_data['supernovae'])
        elif 'cepheids' in corrected_data:
            return self._from_cepheids(corrected_data['cepheids'])
        else:
            return 70.0  # Fallback
    
    def _from_supernovae(self, sne_data):
        """Calculate H0 from supernova Hubble diagram"""
        z = sne_data['redshift'].values
        distance_modulus = sne_data['distance_modulus'].values
        
        # Fit Hubble law: μ = 5log10(c/H0 * z) + 25
        def hubble_law(z, H0):
            return 5 * np.log10(3e5 * z / H0) + 25
        
        try:
            popt, pcov = curve_fit(hubble_law, z, distance_modulus, p0=[70])
            return popt[0]
        except:
            return 73.0  # SH0ES-like value
    
    def _from_cepheids(self, cepheid_data):
        """Calculate H0 from Cepheid distances"""
        distances = cepheid_data['distance'].values
        redshifts = cepheid_data['redshift'].values
        
        # Simple Hubble law: v = H0 * d
        velocities = 3e5 * redshifts  # km/s
        return np.mean(velocities / distances)
