#!/usr/bin/env python
"""
Check that the environment and dependencies are properly set up for the project.
"""
import sys
import os
import importlib
import platform
import subprocess
from pathlib import Path

def check_python_version():
    """Check that Python version is at least 3.8."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"Python version: {sys.version}")
    if current_version < required_version:
        print(f"❌ Python version should be at least {required_version[0]}.{required_version[1]}")
        return False
    else:
        print(f"✅ Python version is compatible")
        return True

def check_dependencies():
    """Check that all required packages are installed."""
    requirements_path = Path(__file__).parent / "requirements.txt"
    
    if not requirements_path.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Parse requirements.txt
    required_packages = []
    with open(requirements_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Extract package name (without version)
                package_name = line.split(">=")[0].split("==")[0].strip()
                required_packages.append(package_name)
    
    # Check each package
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is not installed")
    
    if missing_packages:
        print("\nMissing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✅ All required packages are installed")
        return True

def check_project_structure():
    """Check that the project structure is correct."""
    project_dir = Path(__file__).parent / "project"
    
    if not project_dir.exists():
        print("❌ project/ directory not found")
        return False
    
    expected_dirs = [
        "data",
        "data/audio",
        "data/audio/samples",
        "models",
        "notebooks",
        "src",
        "src/text_analysis",
        "src/speech_analysis",
        "src/llm_integration",
        "src/web_interface"
    ]
    
    missing_dirs = []
    for dir_path in expected_dirs:
        full_path = project_dir / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
            print(f"❌ {dir_path}/ directory not found")
        else:
            print(f"✅ {dir_path}/ directory exists")
    
    if missing_dirs:
        print("\nMissing directories. Create them with:")
        for dir_path in missing_dirs:
            print(f"mkdir -p project/{dir_path}")
        return False
    else:
        print("\n✅ Project structure is correct")
        return True

def check_audio_samples():
    """Check if audio samples are available."""
    samples_dir = Path(__file__).parent / "project" / "data" / "audio" / "samples"
    
    if not samples_dir.exists():
        print("❌ Audio samples directory not found")
        return False
    
    wav_files = list(samples_dir.glob("*.wav"))
    if not wav_files:
        print("❌ No audio samples found")
        print("\nDownload samples with:")
        print("python project/data/audio/download_samples.py")
        return False
    else:
        print(f"✅ {len(wav_files)} audio samples found")
        return True

def main():
    """Run all checks."""
    print("=" * 50)
    print("Environment Check for Speech Sentiment Analysis Project")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Dependencies", check_dependencies),
        ("Audio Samples", check_audio_samples)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name}")
        print("-" * len(name))
        results[name] = check_func()
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        all_passed = all_passed and result
    
    if all_passed:
        print("\n✅ All checks passed. Project environment is ready!")
        return 0
    else:
        print("\n❌ Some checks failed. Please address the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 