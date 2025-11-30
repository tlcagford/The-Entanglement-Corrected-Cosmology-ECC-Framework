#!/usr/bin/env python3
"""
Run comprehensive optimization of ECC framework
"""

import sys
import os
sys.path.append('.')

def main():
    print("🚀 ECC FRAMEWORK PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    try:
        from optimization.parameter_optimizer import ECCParameterOptimizer
        
        # Run optimization
        optimizer = ECCParameterOptimizer()
        results = optimizer.run_comprehensive_optimization()
        
        # Test optimized models
        print("\n🧪 TESTING OPTIMIZED MODELS")
        print("-" * 40)
        
        from theoretical_correction.enhanced_entanglement_models import get_optimized_model
        from theoretical_correction.modified_friedmann_solver import ModifiedFriedmannSolver
        
        solver = ModifiedFriedmannSolver()
        
        for model_name in ['early_dark_energy_like', 'persistent_entanglement', 'quantum_coherence']:
            try:
                model = get_optimized_model(model_name)
                result = solver.solve_with_entanglement(None, model_name, entanglement_model=model)
                
                H0 = result['H0']
                tension_planck = abs(H0 - 67.36) / 0.54
                tension_sh0es = abs(H0 - 73.04) / 1.04
                combined_tension = np.sqrt(tension_planck**2 + tension_sh0es**2)
                
                print(f"   {model_name:25} → H0: {H0:6.2f} | Combined tension: {combined_tension:5.2f}σ")
                
            except Exception as e:
                print(f"   {model_name:25} → ERROR: {e}")
        
        print("\n🎯 OPTIMIZATION COMPLETE!")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("\n💡 Make sure all required modules are installed:")
        print("   pip install scipy matplotlib")

if __name__ == "__main__":
    import numpy as np
    main()
