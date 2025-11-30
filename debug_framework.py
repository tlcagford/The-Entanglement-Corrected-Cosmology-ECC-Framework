#!/usr/bin/env python3
"""
Debug the ECC framework to see what's actually working
"""

import sys
import os
import importlib

def debug_imports():
    """Check if all modules can be imported"""
    modules = [
        'theoretical_correction.entanglement_density_models',
        'theoretical_correction.modified_friedmann_solver', 
        'data_ingestion.hubble_jwst_data_pipeline',
        'tension_resolver.hubble_constant_calculator'
    ]
    
    print("🔧 DEBUGGING ECC FRAMEWORK IMPORTS")
    print("=" * 40)
    
    for module_path in modules:
        try:
            module = importlib.import_module(module_path)
            print(f"✅ {module_path}")
        except Exception as e:
            print(f"❌ {module_path}: {e}")

def check_methods():
    """Check if key methods exist and work"""
    try:
        from theoretical_correction.entanglement_density_models import get_entanglement_model
        from theoretical_correction.modified_friedmann_solver import ModifiedFriedmannSolver
        
        # Test model creation
        model = get_entanglement_model('early_dark_energy_like')
        print(f"✅ Entanglement model created: {model.__class__.__name__}")
        
        # Test density calculation
        density = model.density(0.001)  # a = 0.001
        print(f"✅ Density calculation works: {density:.2e}")
        
        # Test solver
        solver = ModifiedFriedmannSolver()
        print("✅ Friedmann solver initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Method check failed: {e}")
        return False

if __name__ == "__main__":
    debug_imports()
    print()
    check_methods()
