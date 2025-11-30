#!/usr/bin/env python3
"""
Quick test to see if the basic framework works
"""

import sys
import os
sys.path.append('.')

def quick_validation():
    """Run a quick validation of the core components"""
    print("🔍 QUICK VALIDATION OF ECC FRAMEWORK")
    print("=" * 50)
    
    try:
        # Test 1: Import core modules
        from theoretical_correction.entanglement_density_models import get_entanglement_model
        from theoretical_correction.modified_friedmann_solver import ModifiedFriedmannSolver
        
        print("✅ Core modules imported successfully")
        
        # Test 2: Test entanglement models
        solver = ModifiedFriedmannSolver()
        models = ['early_dark_energy_like', 'persistent_entanglement', 'quantum_coherence']
        
        print("\n🧪 Testing Entanglement Models:")
        for model_name in models:
            model = get_entanglement_model(model_name)
            result = solver.solve_with_entanglement(None, model_name)
            print(f"   {model_name:25} → H0 = {result['H0']:.2f}")
        
        # Test 3: Basic tension calculation
        original_tension = 4.8
        avg_h0 = np.mean([solver.solve_with_entanglement(None, m)['H0'] for m in models])
        tension_reduction = original_tension - abs(73.04 - avg_h0) / 1.04
        
        print(f"\n⚖️  Average H0 from models: {avg_h0:.2f}")
        print(f"   Estimated tension reduction: {tension_reduction:.1f}σ")
        
        # Test 4: Theoretical consistency
        h0_values = [solver.solve_with_entanglement(None, m)['H0'] for m in models]
        consistency = 1 - (np.std(h0_values) / np.mean(h0_values))
        
        print(f"\n🎯 Theoretical consistency: {consistency:.1%}")
        
        # Overall assessment
        if tension_reduction > 2.0 and consistency > 0.8:
            print("\n🎉 THEORY SHOWING PROMISING RESULTS!")
            print("   Significant tension reduction with good consistency")
        elif tension_reduction > 1.0:
            print("\n⚠️  THEORY SHOWING MODERATE RESULTS") 
            print("   Some tension reduction, needs refinement")
        else:
            print("\n🔍 THEORY NEEDS SIGNIFICANT WORK")
            print("   Limited tension reduction achieved")
            
        return {
            'average_H0': avg_h0,
            'tension_reduction': tension_reduction,
            'consistency': consistency,
            'models_tested': models
        }
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return None

if __name__ == "__main__":
    results = quick_validation()
    
    if results:
        print(f"\n📊 Quick test completed successfully!")
        print(f"   Next: Run full Bayesian analysis with 'python run_scientific_tests.py'")
