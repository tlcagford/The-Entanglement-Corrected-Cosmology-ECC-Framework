"""
Master test script for ECC Framework
"""
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test ECC Framework')
    parser.add_argument('--path', type=str, default='.',
                       help='Path to ECC framework')
    parser.add_argument('--skip-basic', action='store_true',
                       help='Skip basic tests')
    parser.add_argument('--skip-science', action='store_true',
                       help='Skip scientific tests')
    parser.add_argument('--skip-obs', action='store_true',
                       help='Skip observational tests')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests (default)')
    
    args = parser.parse_args()
    
    # Add ECC to path
    sys.path.insert(0, os.path.abspath(args.path))
    
    results = {}
    
    # Run basic tests
    if not args.skip_basic or args.all:
        print("\n" + "="*70)
        print("RUNNING BASIC TESTS")
        print("="*70)
        try:
            import test_ecc_basic
            basic_passed = test_ecc_basic.run_basic_tests()
            results['basic'] = basic_passed
        except Exception as e:
            print(f"Basic tests failed: {e}")
            results['basic'] = False
    
    # Run scientific tests
    if not args.skip_science or args.all:
        print("\n" + "="*70)
        print("RUNNING SCIENTIFIC TESTS")
        print("="*70)
        try:
            import test_ecc_scientific
            # This requires importing the actual ECC framework
            import ecc  # or whatever the module is called
            science_passed = test_ecc_scientific.run_scientific_tests(ecc)
            results['science'] = science_passed
        except Exception as e:
            print(f"Scientific tests failed: {e}")
            results['science'] = False
    
    # Run observational tests
    if not args.skip_obs or args.all:
        print("\n" + "="*70)
        print("RUNNING OBSERVATIONAL TESTS")
        print("="*70)
        try:
            import test_ecc_observations
            import ecc
            obs_passed = test_ecc_observations.run_observational_tests(ecc)
            results['observational'] = obs_passed
        except Exception as e:
            print(f"Observational tests failed: {e}")
            results['observational'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_type, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_type.upper():15} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} test suites passed")
    
    if total_passed == total_tests:
        print("\n✓ ECC framework passed all tests!")
        return 0
    else:
        print("\n✗ ECC framework failed some tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())
