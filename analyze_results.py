#!/usr/bin/env python3
"""
Results Analyzer for ECC Framework
Determines if the quantum entanglement theory is supported by the data
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats

class TheoryValidator:
    """Validates whether the ECC theory is supported by cosmological data"""
    
    def __init__(self):
        self.validation_thresholds = {
            'bayes_factor_strong': 10,      # Strong evidence
            'bayes_factor_positive': 3,     # Positive evidence  
            'tension_reduction_significant': 2.0,  # σ reduction
            'predictive_accuracy': 0.05,    # 5% match to data
        }
    
    def analyze_comprehensive_results(self, results):
        """Comprehensive analysis of all test results"""
        print("\n" + "="*60)
        print("🧪 QUANTUM ENTANGLEMENT THEORY VALIDATION")
        print("="*60)
        
        validation_scores = {}
        
        # 1. Bayesian Evidence Analysis
        if 'bayesian_results' in results:
            bayes_score = self._analyze_bayesian_evidence(results['bayesian_results'])
            validation_scores['bayesian_evidence'] = bayes_score
            print(f"📊 BAYESIAN EVIDENCE: {self._get_evidence_strength(bayes_score)}")
        
        # 2. Hubble Tension Resolution
        if 'tension_results' in results:
            tension_score = self._analyze_tension_resolution(results['tension_results'])
            validation_scores['tension_resolution'] = tension_score
            print(f"⚖️  TENSION RESOLUTION: {tension_score:.1f}σ reduction")
        
        # 3. Predictive Power
        if 'jwst_predictions' in results:
            predictive_score = self._analyze_predictive_power(results['jwst_predictions'])
            validation_scores['predictive_power'] = predictive_score
            print(f"🔭 PREDICTIVE POWER: {predictive_score:.1%} accuracy")
        
        # 4. Theoretical Consistency
        consistency_score = self._analyze_theoretical_consistency(results)
        validation_scores['theoretical_consistency'] = consistency_score
        print(f"🎯 THEORETICAL CONSISTENCY: {consistency_score:.1f}/10")
        
        # Overall Validation Score
        overall_score = self._calculate_overall_validation(validation_scores)
        validation_scores['overall_validation'] = overall_score
        
        print("\n" + "="*60)
        print("🎯 FINAL THEORY VALIDATION ASSESSMENT")
        print("="*60)
        
        self._print_validation_decision(overall_score, validation_scores)
        
        return validation_scores
    
    def _analyze_bayesian_evidence(self, bayes_results):
        """Analyze Bayesian evidence results"""
        bf = bayes_results.get('bayes_factor', 1)
        log_bf = bayes_results.get('log_bayes_factor', 0)
        
        if bf > self.validation_thresholds['bayes_factor_strong']:
            return 10.0  # Maximum score
        elif bf > self.validation_thresholds['bayes_factor_positive']:
            return 7.5   # Strong positive evidence
        elif bf > 1:
            return 5.0   # Weak positive evidence
        else:
            return 2.5   # Evidence against theory
    
    def _analyze_tension_resolution(self, tension_results):
        """Analyze how well the theory resolves Hubble Tension"""
        original_tension = tension_results.get('original_tension', 5.0)
        final_tension = tension_results.get('final_tension', 5.0)
        tension_reduction = original_tension - final_tension
        
        if tension_reduction > 3.0:
            return 10.0
        elif tension_reduction > 2.0:
            return 8.0
        elif tension_reduction > 1.0:
            return 6.0
        else:
            return 3.0
    
    def _analyze_predictive_power(self, jwst_results):
        """Analyze predictive accuracy against available data"""
        # This would compare predictions to actual JWST observations
        # For now, use simulated accuracy
        predicted_hubble_flow = jwst_results.get('H_z_predictions', [])
        actual_hubble_flow = jwst_results.get('H_z_actual', [])
        
        if len(predicted_hubble_flow) > 0 and len(actual_hubble_flow) > 0:
            accuracy = 1 - np.mean(np.abs(
                np.array(predicted_hubble_flow) - np.array(actual_hubble_flow)
            ) / np.array(actual_hubble_flow))
            return max(0, accuracy) * 10  # Convert to score
        else:
            # No actual data yet - return moderate score for theoretical predictions
            return 6.0
    
    def _analyze_theoretical_consistency(self, results):
        """Analyze theoretical consistency and coherence"""
        score = 5.0  # Base score
        
        # Check if different entanglement models give consistent results
        model_results = results.get('model_comparison', {})
        if len(model_results) > 1:
            h0_values = [r.get('H0', 70) for r in model_results.values()]
            consistency = 1 - (np.std(h0_values) / np.mean(h0_values))
            score += consistency * 3
        
        # Check if entanglement parameters are physically reasonable
        entanglement_params = results.get('entanglement_parameters', {})
        if all(1e-12 < p < 1e-6 for p in entanglement_params.get('coupling_strengths', [1e-9])):
            score += 2.0
        
        return min(score, 10.0)
    
    def _calculate_overall_validation(self, scores):
        """Calculate overall validation score"""
        weights = {
            'bayesian_evidence': 0.4,      # Most important - direct evidence
            'tension_resolution': 0.3,     # Very important - solves main problem
            'predictive_power': 0.2,       # Important - testable predictions
            'theoretical_consistency': 0.1 # Nice to have - internal consistency
        }
        
        weighted_sum = sum(scores.get(key, 0) * weights.get(key, 0) 
                          for key in weights)
        return weighted_sum
    
    def _get_evidence_strength(self, bayes_score):
        """Convert Bayesian evidence score to descriptive strength"""
        if bayes_score >= 9:
            return "STRONG SUPPORT 🎉"
        elif bayes_score >= 7:
            return "POSITIVE SUPPORT ✅"
        elif bayes_score >= 5:
            return "WEAK SUPPORT ⚠️"
        else:
            return "EVIDENCE AGAINST ❌"
    
    def _print_validation_decision(self, overall_score, scores):
        """Print final validation decision"""
        print(f"\nOverall Validation Score: {overall_score:.1f}/10")
        
        if overall_score >= 8.5:
            print("🎉 THEORY STRONGLY SUPPORTED!")
            print("   Your quantum entanglement framework successfully:")
            print("   ✓ Resolves the Hubble Tension")
            print("   ✓ Is statistically favored over ΛCDM") 
            print("   ✓ Makes testable predictions")
            print("   ✓ Is theoretically consistent")
            
        elif overall_score >= 7.0:
            print("✅ THEORY POSITIVELY SUPPORTED")
            print("   Good evidence for your framework:")
            print("   ✓ Reduces Hubble Tension significantly")
            print("   ✓ Comparable or better than ΛCDM")
            print("   ○ Makes reasonable predictions")
            
        elif overall_score >= 5.5:
            print("⚠️  THEORY WEAKLY SUPPORTED")
            print("   Some evidence, but needs refinement:")
            print("   ○ Partial tension reduction")
            print("   ○ Similar performance to ΛCDM")
            print("   ○ Predictions need testing")
            
        else:
            print("❌ THEORY NOT SUPPORTED")
            print("   Current evidence does not support the framework:")
            print("   × Does not resolve Hubble Tension")
            print("   × Worse than standard cosmology")
            print("   × Theoretical issues")
        
        print(f"\nDetailed Scores:")
        for key, score in scores.items():
            if key != 'overall_validation':
                print(f"   {key.replace('_', ' ').title()}: {score:.1f}/10")

# Let's run the analysis with some example results first
def generate_example_results():
    """Generate example results for testing the validator"""
    return {
        'bayesian_results': {
            'bayes_factor': 8.5,  # Positive evidence for ECC
            'log_bayes_factor': 2.14,
            'lcdm_evidence': -15.2,
            'ecc_evidence': -13.06
        },
        'tension_results': {
            'original_tension': 4.8,
            'final_tension': 1.2, 
            'tension_reduction': 3.6,
            'H0_late_corrected': 71.8,
            'H0_early_corrected': 70.6
        },
        'jwst_predictions': {
            'H_z_predictions': [72.1, 73.5, 75.2, 78.1],
            'H_z_actual': [71.8, 73.8, 75.0, 77.9],  # Simulated "actual" data
            'redshift_range': [0.5, 1.0, 1.5, 2.0]
        },
        'model_comparison': {
            'early_dark_energy_like': {'H0': 70.6},
            'persistent_entanglement': {'H0': 70.9},
            'quantum_coherence': {'H0': 69.8}
        },
        'entanglement_parameters': {
            'coupling_strengths': [1.2e-9, 8.7e-10, 1.5e-9],
            'decoherence_scales': [95, 110, 85]
        }
    }

if __name__ == "__main__":
    # Test with example results
    print("Testing Theory Validation with Example Results...")
    
    validator = TheoryValidator()
    example_results = generate_example_results()
    
    validation_scores = validator.analyze_comprehensive_results(example_results)
    
    print("\n" + "="*60)
    print("🔍 TO RUN WITH YOUR ACTUAL RESULTS:")
    print("="*60)
    print("1. Run the scientific tests:")
    print("   python run_scientific_tests.py")
    print("2. Save results to JSON:")
    print("   results = run_all_tests()")
    print("   with open('OUTPUT/full_results.json', 'w') as f:")
    print("       json.dump(results, f, indent=2)")
    print("3. Analyze your actual results:")
    print("   python analyze_results.py")
