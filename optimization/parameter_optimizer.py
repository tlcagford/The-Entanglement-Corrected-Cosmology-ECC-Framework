#!/usr/bin/env python3
"""
Parameter Optimization for ECC Framework
Systematically finds optimal entanglement parameters to resolve Hubble Tension
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import matplotlib.pyplot as plt

class ECCParameterOptimizer:
    def __init__(self):
        # Target: Bridge between Planck (67.4) and SH0ES (73.0)
        self.target_H0 = 70.2  # Initial target - will optimize this too
        self.planck_H0 = 67.36
        self.sh0es_H0 = 73.04
        self.optimal_results = {}
    
    def objective_function(self, params, model_type='early_dark_energy_like'):
        """
        Objective: Minimize Hubble Tension
        Lower score = better tension resolution
        """
        try:
            from theoretical_correction.entanglement_density_models import get_entanglement_model
            from theoretical_correction.modified_friedmann_solver import ModifiedFriedmannSolver
            
            # Unpack parameters based on model type
            if model_type == 'early_dark_energy_like':
                coupling, peak_redshift, peak_fraction, decay_width = params
                model = get_entanglement_model(
                    model_type,
                    coupling_strength=10**coupling,  # Log scale
                    peak_redshift=peak_redshift,
                    peak_fraction=peak_fraction,
                    decay_width=decay_width
                )
            elif model_type == 'persistent_entanglement':
                coupling, primordial_strength, decay_index = params
                model = get_entanglement_model(
                    model_type,
                    coupling_strength=10**coupling,
                    primordial_strength=10**primordial_strength,
                    decay_index=decay_index
                )
            else:  # quantum_coherence
                coupling, coherence_length, quantum_fluctuations = params
                model = get_entanglement_model(
                    model_type,
                    coupling_strength=10**coupling,
                    coherence_length=coherence_length,
                    quantum_fluctuations=10**quantum_fluctuations
                )
            
            # Solve for H0
            solver = ModifiedFriedmannSolver()
            result = solver.solve_with_entanglement(None, model_type, 
                                                  entanglement_model=model)
            H0_pred = result['H0']
            
            # Calculate tension with both datasets
            tension_planck = abs(H0_pred - self.planck_H0) / 0.54
            tension_sh0es = abs(H0_pred - self.sh0es_H0) / 1.04
            
            # Combined tension (what we want to minimize)
            combined_tension = np.sqrt(tension_planck**2 + tension_sh0es**2)
            
            # Penalty for unphysical parameters
            penalty = 0
            if H0_pred < 60 or H0_pred > 80:
                penalty += 100
            if combined_tension > 10:  # Really bad fits
                penalty += 50
                
            return combined_tension + penalty
            
        except Exception as e:
            return 1000  # Large penalty for failed calculations
    
    def optimize_model(self, model_type, n_iterations=3):
        """Optimize parameters for a specific entanglement model"""
        print(f"\n🔧 OPTIMIZING {model_type.upper().replace('_', ' ')}")
        print("-" * 50)
        
        # Parameter bounds (log scale for small numbers)
        if model_type == 'early_dark_energy_like':
            bounds = [
                (-12, -6),    # coupling_strength (log scale)
                (500, 2000),  # peak_redshift
                (0.01, 0.2),  # peak_fraction
                (50, 300)     # decay_width
            ]
            initial_guess = [-9, 1100, 0.05, 100]
            
        elif model_type == 'persistent_entanglement':
            bounds = [
                (-12, -6),    # coupling_strength
                (-10, -5),    # primordial_strength (log scale)
                (0.1, 2.0)    # decay_index
            ]
            initial_guess = [-9, -8, 0.5]
            
        else:  # quantum_coherence
            bounds = [
                (-12, -6),    # coupling_strength
                (100, 5000),  # coherence_length
                (-8, -3)      # quantum_fluctuations (log scale)
            ]
            initial_guess = [-9, 1000, -5]
        
        best_result = None
        best_score = float('inf')
        
        # Try multiple optimization methods
        methods = [
            ('differential_evolution', self._optimize_de),
            ('nelder-mead', self._optimize_nm),
            ('powell', self._optimize_powell)
        ]
        
        for method_name, optimizer in methods:
            print(f"   Trying {method_name}...")
            try:
                result = optimizer(bounds, model_type)
                if result.fun < best_score:
                    best_score = result.fun
                    best_result = result
                    print(f"     → Improved score: {best_score:.3f}")
            except Exception as e:
                print(f"     → {method_name} failed: {e}")
        
        if best_result is None:
            print("   ❌ All optimization methods failed")
            return None
        
        # Get the optimized H0 value
        optimized_H0 = self._calculate_optimized_H0(best_result.x, model_type)
        
        print(f"   ✅ Optimized H0: {optimized_H0:.2f}")
        print(f"   📊 Final tension score: {best_score:.3f}")
        
        self.optimal_results[model_type] = {
            'parameters': best_result.x,
            'H0': optimized_H0,
            'tension_score': best_score,
            'bounds': bounds
        }
        
        return self.optimal_results[model_type]
    
    def _optimize_de(self, bounds, model_type):
        """Differential evolution - good for global optimization"""
        result = differential_evolution(
            lambda x: self.objective_function(x, model_type),
            bounds,
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=1e-4,
            disp=False
        )
        return result
    
    def _optimize_nm(self, bounds, model_type):
        """Nelder-Mead - good for local refinement"""
        # Start from center of bounds
        x0 = np.mean(bounds, axis=1)
        result = minimize(
            lambda x: self.objective_function(x, model_type),
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 200, 'disp': False}
        )
        return result
    
    def _optimize_powell(self, bounds, model_type):
        """Powell's method - good for nonlinear problems"""
        x0 = np.mean(bounds, axis=1)
        result = minimize(
            lambda x: self.objective_function(x, model_type),
            x0,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 200, 'disp': False}
        )
        return result
    
    def _calculate_optimized_H0(self, params, model_type):
        """Calculate H0 from optimized parameters"""
        try:
            from theoretical_correction.entanglement_density_models import get_entanglement_model
            from theoretical_correction.modified_friedmann_solver import ModifiedFriedmannSolver
            
            if model_type == 'early_dark_energy_like':
                coupling, peak_redshift, peak_fraction, decay_width = params
                model = get_entanglement_model(
                    model_type,
                    coupling_strength=10**coupling,
                    peak_redshift=peak_redshift,
                    peak_fraction=peak_fraction,
                    decay_width=decay_width
                )
            elif model_type == 'persistent_entanglement':
                coupling, primordial_strength, decay_index = params
                model = get_entanglement_model(
                    model_type,
                    coupling_strength=10**coupling,
                    primordial_strength=10**primordial_strength,
                    decay_index=decay_index
                )
            else:
                coupling, coherence_length, quantum_fluctuations = params
                model = get_entanglement_model(
                    model_type,
                    coupling_strength=10**coupling,
                    coherence_length=coherence_length,
                    quantum_fluctuations=10**quantum_fluctuations
                )
            
            solver = ModifiedFriedmannSolver()
            result = solver.solve_with_entanglement(None, model_type, entanglement_model=model)
            return result['H0']
            
        except Exception as e:
            print(f"Error calculating optimized H0: {e}")
            return 70.0
    
    def run_comprehensive_optimization(self):
        """Optimize all three entanglement models"""
        print("🚀 COMPREHENSIVE PARAMETER OPTIMIZATION")
        print("=" * 60)
        
        models = ['early_dark_energy_like', 'persistent_entanglement', 'quantum_coherence']
        
        for model in models:
            self.optimize_model(model)
        
        self._compare_optimized_models()
        self._plot_optimization_results()
        
        return self.optimal_results
    
    def _compare_optimized_models(self):
        """Compare performance of optimized models"""
        print("\n📊 OPTIMIZATION RESULTS COMPARISON")
        print("-" * 50)
        
        if not self.optimal_results:
            print("   No successful optimizations")
            return
        
        best_model = None
        best_score = float('inf')
        
        for model_name, results in self.optimal_results.items():
            if results['tension_score'] < best_score:
                best_score = results['tension_score']
                best_model = model_name
            
            tension_planck = abs(results['H0'] - self.planck_H0) / 0.54
            tension_sh0es = abs(results['H0'] - self.sh0es_H0) / 1.04
            
            print(f"   {model_name:25} → H0: {results['H0']:6.2f} | "
                  f"Tension: {results['tension_score']:5.2f} | "
                  f"Planck: {tension_planck:4.1f}σ, SH0ES: {tension_sh0es:4.1f}σ")
        
        print(f"\n   🏆 BEST MODEL: {best_model}")
        print(f"   🎯 Best H0: {self.optimal_results[best_model]['H0']:.2f}")
        print(f"   📉 Best tension score: {best_score:.3f}")
        
        # Calculate tension reduction
        original_tension = abs(self.sh0es_H0 - self.planck_H0) / np.sqrt(1.04**2 + 0.54**2)
        best_tension = self.optimal_results[best_model]['tension_score']
        reduction = original_tension - best_tension
        
        print(f"   📈 Tension reduction: {reduction:+.2f}σ")
        
        if reduction > 2:
            print("   🎉 EXCELLENT TENSION RESOLUTION ACHIEVED!")
        elif reduction > 1:
            print("   ✅ GOOD TENSION REDUCTION")
        elif reduction > 0:
            print("   ⚠️  MODEST TENSION REDUCTION")
        else:
            print("   ❌ TENSION INCREASED")
    
    def _plot_optimization_results(self):
        """Plot optimization results"""
        if not self.optimal_results:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot H0 values
        models = list(self.optimal_results.keys())
        h0_values = [results['H0'] for results in self.optimal_results.values()]
        tension_scores = [results['tension_score'] for results in self.optimal_results.values()]
        
        # Bar plot of H0 values
        bars = ax1.bar(models, h0_values, alpha=0.7, color=['blue', 'green', 'red'])
        ax1.axhline(y=self.planck_H0, color='red', linestyle='--', alpha=0.7, label='Planck')
        ax1.axhline(y=self.sh0es_H0, color='orange', linestyle='--', alpha=0.7, label='SH0ES')
        ax1.axhline(y=70.2, color='purple', linestyle='--', alpha=0.7, label='Target')
        ax1.set_ylabel('H₀ (km/s/Mpc)')
        ax1.set_title('Optimized H₀ Values')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, h0 in zip(bars, h0_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{h0:.2f}', ha='center', va='bottom')
        
        # Plot tension scores
        ax2.bar(models, tension_scores, alpha=0.7, color=['blue', 'green', 'red'])
        ax2.set_ylabel('Combined Tension (σ)')
        ax2.set_title('Optimized Tension Scores')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, score in enumerate(tension_scores):
            ax2.text(i, score + 0.1, f'{score:.2f}σ', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('OUTPUT/optimization_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📈 Optimization plots saved to: OUTPUT/optimization_results.png")

def main():
    """Run the optimization"""
    optimizer = ECCParameterOptimizer()
    results = optimizer.run_comprehensive_optimization()
    
    # Save optimized parameters
    import json
    with open('OUTPUT/optimized_parameters.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"\n💾 Optimized parameters saved to: OUTPUT/optimized_parameters.json")
    
    return results

if __name__ == "__main__":
    main()
