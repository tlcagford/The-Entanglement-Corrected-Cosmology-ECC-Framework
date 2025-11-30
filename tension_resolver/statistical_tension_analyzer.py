import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class TensionAnalyzer:
    """Analyze the reduction in Hubble Tension after corrections"""
    
    def __init__(self):
        self.original_H0_late = 73.0  # SH0ES value
        self.original_H0_early = 67.4  # Planck value
        self.original_error_late = 1.0
        self.original_error_early = 0.5
        
    def calculate_tension(self, H0_late_corrected, H0_early_corrected, 
                         error_late=0.8, error_early=0.4):
        """
        Calculate tension between late and early universe measurements
        Returns tension in sigma
        """
        # Difference between measurements
        delta_H0 = H0_late_corrected - H0_early_corrected
        
        # Combined uncertainty
        combined_error = np.sqrt(error_late**2 + error_early**2)
        
        # Tension in sigma
        tension_sigma = abs(delta_H0) / combined_error
        
        # Calculate percentage reduction in tension
        original_delta = self.original_H0_late - self.original_H0_early
        original_combined_error = np.sqrt(self.original_error_late**2 + self.original_error_early**2)
        original_tension = original_delta / original_combined_error
        
        tension_reduction = (1 - tension_sigma / original_tension) * 100
        
        return {
            'sigma_tension': tension_sigma,
            'original_tension': original_tension,
            'tension_reduction_percent': tension_reduction,
            'H0_difference': delta_H0,
            'combined_uncertainty': combined_error
        }
    
    def plot_tension_evolution(self, results):
        """Plot the evolution of Hubble Tension through correction steps"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot H0 values
        methods = ['Original Late', 'Original Early', 'Corrected Late', 'Corrected Early']
        h0_values = [self.original_H0_late, self.original_H0_early,
                    results['h0_late_corrected'], results['h0_early_corrected']]
        errors = [self.original_error_late, self.original_error_early, 0.8, 0.4]
        
        ax1.errorbar(range(len(methods)), h0_values, yerr=errors, 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45)
        ax1.set_ylabel('H₀ (km/s/Mpc)')
        ax1.set_title('Hubble Constant Measurements')
        ax1.grid(True, alpha=0.3)
        
        # Plot tension reduction
        original_tension = results['tension_analysis']['original_tension']
        final_tension = results['tension_analysis']['sigma_tension']
        
        ax2.bar(['Original', 'ECC Corrected'], [original_tension, final_tension], 
               color=['red', 'green'], alpha=0.7)
        ax2.set_ylabel('Tension (σ)')
        ax2.set_title('Hubble Tension Reduction')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('OUTPUT/tension_evolution_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Tension evolution plot saved to OUTPUT/tension_evolution_plots.png")
