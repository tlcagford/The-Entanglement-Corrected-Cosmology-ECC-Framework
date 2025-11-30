#!/usr/bin/env python3
"""
ENTANGLEMENT-CORRECTED COSMOLOGY (ECC) FRAMEWORK
Orchestrator Module

Author: tlcagford
License: MIT
Description: Main orchestrator for resolving the Hubble Tension through
             quantum entanglement corrections to both observational data
             and cosmological models.
"""

import logging
import numpy as np
from pathlib import Path
import json

# Import project modules
from data_ingestion.hubble_jwst_data_pipeline import DataPipeline
from observational_correction.quantum_psf_correction import QuantumPSFCorrector
from observational_correction.entanglement_luminosity_correction import EntanglementLuminosityCorrector
from theoretical_correction.modified_friedmann_solver import ModifiedFriedmannSolver
from theoretical_correction.entanglement_density_models import EntanglementDensityModels
from tension_resolver.hubble_constant_calculator import HubbleConstantCalculator
from tension_resolver.statistical_tension_analyzer import TensionAnalyzer

class ECCOrchestrator:
    def __init__(self, config_path="config/ecc_config.json"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.results = {}
        
        # Initialize modules
        self.data_pipeline = DataPipeline(self.config['data_sources'])
        self.psf_corrector = QuantumPSFCorrector(self.config['psf_params'])
        self.luminosity_corrector = EntanglementLuminosityCorrector(
            self.config['entanglement_params'])
        self.friedmann_solver = ModifiedFriedmannSolver()
        self.hubble_calculator = HubbleConstantCalculator()
        self.tension_analyzer = TensionAnalyzer()
        
        logging.info("🎯 ECC Framework Initialized")

    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("Config file not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self):
        """Return default configuration"""
        return {
            'data_sources': {
                'hubble_cepheids': 'data/raw/hubble_cepheids.csv',
                'jwst_supernovae': 'data/raw/jwst_supernovae.csv',
                'planck_cmb': 'data/raw/planck_spectra.fits'
            },
            'psf_params': {
                'quantum_deconvolution': True,
                'neural_enhancement': True,
                'entanglement_aware': True
            },
            'entanglement_params': {
                'dark_photon_coupling': 1e-9,
                'decoherence_scale': 100,  # Mpc
                'correction_model': 'wavefunction_collapse'
            }
        }

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ecc_framework.log'),
                logging.StreamHandler()
            ]
        )

    def run_full_analysis(self):
        """
        Execute the complete ECC analysis pipeline
        """
        logging.info("🚀 Starting Full ECC Analysis Pipeline")
        
        try:
            # Phase 1: Observational Correction
            h0_late_corrected = self.run_observational_correction()
            self.results['h0_late_corrected'] = h0_late_corrected
            
            # Phase 2: Theoretical Correction  
            h0_early_corrected = self.run_theoretical_correction()
            self.results['h0_early_corrected'] = h0_early_corrected
            
            # Phase 3: Tension Resolution
            tension_result = self.analyze_tension()
            self.results['tension_analysis'] = tension_result
            
            self.save_results()
            logging.info("✅ ECC Analysis Complete")
            
            return self.results
            
        except Exception as e:
            logging.error(f"❌ Analysis failed: {str(e)}")
            raise

    def run_observational_correction(self):
        """
        Phase 1: Correct late-universe measurements using quantum-aware image processing
        """
        logging.info("🔭 Starting Observational Correction Phase")
        
        # Load raw observational data
        raw_data = self.data_pipeline.load_observational_data()
        
        # Apply quantum PSF correction (from Astronomical-Image-Refiner)
        psf_corrected = self.psf_corrector.apply_correction(raw_data)
        
        # Apply entanglement luminosity correction
        entanglement_corrected = self.luminosity_corrector.apply_correction(psf_corrected)
        
        # Calculate corrected H0 value
        h0_corrected = self.hubble_calculator.calculate_from_distance_ladder(
            entanglement_corrected)
            
        logging.info(f"📏 Corrected Late-Universe H0: {h0_corrected:.2f} km/s/Mpc")
        return h0_corrected

    def run_theoretical_correction(self):
        """
        Phase 2: Correct early-universe prediction using entanglement-modified cosmology
        """
        logging.info("📐 Starting Theoretical Correction Phase")
        
        # Load CMB data
        cmb_data = self.data_pipeline.load_cmb_data()
        
        # Solve modified Friedmann equations with entanglement term
        cosmological_params = self.friedmann_solver.solve_with_entanglement(
            cmb_data, 
            model_name='early_dark_energy_like'
        )
        
        # Extract corrected H0 prediction
        h0_corrected = cosmological_params['H0']
        
        logging.info(f"🌌 Corrected Early-Universe H0: {h0_corrected:.2f} km/s/Mpc")
        return h0_corrected

    def analyze_tension(self):
        """
        Phase 3: Analyze reduction in Hubble Tension
        """
        logging.info("⚖️ Analyzing Tension Reduction")
        
        tension_result = self.tension_analyzer.calculate_tension(
            self.results['h0_late_corrected'],
            self.results['h0_early_corrected']
        )
        
        logging.info(f"📊 Final Tension: {tension_result['sigma_tension']:.2f}σ")
        return tension_result

    def save_results(self):
        """Save all results to output directory"""
        output_dir = Path("OUTPUT")
        output_dir.mkdir(exist_ok=True)
        
        # Save results as JSON
        with open(output_dir / "ecc_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate tension evolution plot
        self.tension_analyzer.plot_tension_evolution(self.results)

if __name__ == "__main__":
    # Initialize and run the framework
    orchestrator = ECCOrchestrator()
    results = orchestrator.run_full_analysis()
    
    print("\n" + "="*50)
    print("🎉 ECC FRAMEWORK RESULTS")
    print("="*50)
    print(f"Late-Universe H0 (corrected): {results['h0_late_corrected']:.2f}")
    print(f"Early-Universe H0 (corrected): {results['h0_early_corrected']:.2f}") 
    print(f"Remaining Tension: {results['tension_analysis']['sigma_tension']:.2f}σ")
    print("="*50)
