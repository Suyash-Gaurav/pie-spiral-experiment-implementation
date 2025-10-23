#!/usr/bin/env python3
"""
Package Verification Script

Verifies that the π-Spiral Positional Encoding package is complete and ready for use.
Checks for:
- All required files and directories
- Proper __init__.py files
- Configuration files
- Documentation
- Examples
- Tests
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if os.path.isdir(dirpath):
        num_files = len(list(Path(dirpath).rglob('*')))
        print(f"✓ {description}: {dirpath} ({num_files} items)")
        return True
    else:
        print(f"✗ {description}: {dirpath} - MISSING")
        return False

def main():
    print("=" * 80)
    print("π-Spiral Positional Encoding Package Verification")
    print("=" * 80)
    
    base_dir = "./final"
    all_checks = []
    
    # Check main documentation
    print("\n" + "=" * 80)
    print("Main Documentation")
    print("=" * 80)
    all_checks.append(check_file_exists(f"{base_dir}/README.md", "Main README"))
    all_checks.append(check_file_exists(f"{base_dir}/requirements.txt", "Requirements"))
    all_checks.append(check_file_exists(f"{base_dir}/CHANGELOG.md", "Changelog"))
    all_checks.append(check_file_exists(f"{base_dir}/ARCHITECTURE.md", "Architecture"))
    all_checks.append(check_file_exists(f"{base_dir}/EXPERIMENTS.md", "Experiments Guide"))
    
    # Check source code structure
    print("\n" + "=" * 80)
    print("Source Code Structure")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/src", "Source directory"))
    all_checks.append(check_file_exists(f"{base_dir}/src/__init__.py", "Package init"))
    all_checks.append(check_file_exists(f"{base_dir}/src/config.py", "Config module"))
    all_checks.append(check_file_exists(f"{base_dir}/src/data_utils.py", "Data utils"))
    
    # Check encodings
    print("\n" + "=" * 80)
    print("Positional Encodings")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/src/encodings", "Encodings directory"))
    all_checks.append(check_file_exists(f"{base_dir}/src/encodings/__init__.py", "Encodings init"))
    all_checks.append(check_file_exists(f"{base_dir}/src/encodings/pi_spiral.py", "π-Spiral encoding"))
    all_checks.append(check_file_exists(f"{base_dir}/src/encodings/rope.py", "RoPE encoding"))
    all_checks.append(check_file_exists(f"{base_dir}/src/encodings/hybrid.py", "Hybrid encoding"))
    
    # Check models
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/src/models", "Models directory"))
    all_checks.append(check_file_exists(f"{base_dir}/src/models/__init__.py", "Models init"))
    all_checks.append(check_file_exists(f"{base_dir}/src/models/transformer.py", "Transformer model"))
    all_checks.append(check_file_exists(f"{base_dir}/src/models/attention.py", "Attention module"))
    all_checks.append(check_file_exists(f"{base_dir}/src/models/adapters.py", "Model adapters"))
    
    # Check training infrastructure
    print("\n" + "=" * 80)
    print("Training Infrastructure")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/src/training", "Training directory"))
    all_checks.append(check_file_exists(f"{base_dir}/src/training/__init__.py", "Training init"))
    all_checks.append(check_file_exists(f"{base_dir}/src/training/trainer.py", "Trainer"))
    all_checks.append(check_file_exists(f"{base_dir}/src/training/logger.py", "Logger"))
    
    # Check diagnostics
    print("\n" + "=" * 80)
    print("Diagnostic Tools")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/src/diagnostics", "Diagnostics directory"))
    all_checks.append(check_file_exists(f"{base_dir}/src/diagnostics/__init__.py", "Diagnostics init"))
    all_checks.append(check_file_exists(f"{base_dir}/src/diagnostics/encoding_analysis.py", "Encoding analysis"))
    all_checks.append(check_file_exists(f"{base_dir}/src/diagnostics/attention_viz.py", "Attention viz"))
    all_checks.append(check_file_exists(f"{base_dir}/src/diagnostics/profiling.py", "Profiling"))
    all_checks.append(check_file_exists(f"{base_dir}/src/diagnostics/visualization.py", "Visualization"))
    
    # Check tests
    print("\n" + "=" * 80)
    print("Test Suite")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/tests", "Tests directory"))
    all_checks.append(check_file_exists(f"{base_dir}/tests/__init__.py", "Tests init"))
    all_checks.append(check_file_exists(f"{base_dir}/tests/test_encodings.py", "Encoding tests"))
    all_checks.append(check_file_exists(f"{base_dir}/tests/test_niah.py", "NIAH tests"))
    all_checks.append(check_file_exists(f"{base_dir}/tests/test_ruler.py", "RULER tests"))
    all_checks.append(check_file_exists(f"{base_dir}/tests/test_infinitebench.py", "InfiniteBench tests"))
    
    # Check scripts
    print("\n" + "=" * 80)
    print("Executable Scripts")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/scripts", "Scripts directory"))
    all_checks.append(check_file_exists(f"{base_dir}/scripts/train.py", "Training script"))
    all_checks.append(check_file_exists(f"{base_dir}/scripts/evaluate.py", "Evaluation script"))
    all_checks.append(check_file_exists(f"{base_dir}/scripts/diagnose.py", "Diagnostic script"))
    
    # Check configurations
    print("\n" + "=" * 80)
    print("Configuration Files")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/configs", "Configs directory"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/README.md", "Configs README"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/base_config.yaml", "Base config"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/qwen_1.5b_pi_spiral.yaml", "Qwen 1.5B config"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/llama_8b_pi_spiral.yaml", "Llama 8B config"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/llama_34b_pi_spiral.yaml", "Llama 34B config"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/baseline_rope.yaml", "RoPE baseline"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/baseline_rope_ntk.yaml", "RoPE-NTK baseline"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/baseline_alibi.yaml", "ALiBi baseline"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/ablation_irrational.yaml", "Irrational ablation"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/ablation_attractor.yaml", "Attractor ablation"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/ablation_alpha_policy.yaml", "Alpha policy ablation"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/ablation_hybrid_k.yaml", "Hybrid K ablation"))
    all_checks.append(check_file_exists(f"{base_dir}/configs/phase2_sanity.yaml", "Phase 2 config"))
    
    # Check examples
    print("\n" + "=" * 80)
    print("Example Scripts")
    print("=" * 80)
    all_checks.append(check_directory_exists(f"{base_dir}/examples", "Examples directory"))
    all_checks.append(check_file_exists(f"{base_dir}/examples/README.md", "Examples README"))
    all_checks.append(check_file_exists(f"{base_dir}/examples/example_train_simple.py", "Simple training example"))
    all_checks.append(check_file_exists(f"{base_dir}/examples/example_evaluate_niah.py", "NIAH evaluation example"))
    all_checks.append(check_file_exists(f"{base_dir}/examples/example_diagnostic.py", "Diagnostic example"))
    all_checks.append(check_file_exists(f"{base_dir}/examples/example_custom_encoding.py", "Custom encoding example"))
    all_checks.append(check_file_exists(f"{base_dir}/examples/example_pretrained_adapter.py", "Pretrained adapter example"))
    
    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    
    total_checks = len(all_checks)
    passed_checks = sum(all_checks)
    failed_checks = total_checks - passed_checks
    
    print(f"\nTotal checks: {total_checks}")
    print(f"Passed: {passed_checks} ✓")
    print(f"Failed: {failed_checks} ✗")
    
    if failed_checks == 0:
        print("\n" + "=" * 80)
        print("✓ Package verification PASSED!")
        print("=" * 80)
        print("\nThe π-Spiral Positional Encoding package is complete and ready for use.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r final/requirements.txt")
        print("  2. Run tests: pytest final/tests/")
        print("  3. Try examples: python final/examples/example_train_simple.py")
        print("  4. Read documentation: final/README.md")
        return 0
    else:
        print("\n" + "=" * 80)
        print("✗ Package verification FAILED!")
        print("=" * 80)
        print(f"\n{failed_checks} file(s) or directory(ies) are missing.")
        print("Please check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
