
import subprocess
import os

# Create a temporary directory for building the wheel
build_dir = r"C:\Users\Damja\CODING_LOCAL\Kaggle\wheels"
os.makedirs(build_dir, exist_ok=True)

# Install wheel package if not already installed
subprocess.check_call(["pip", "install", "wheel"])

# Build the wheel with dependencies
subprocess.check_call([
    "pip", "wheel",
    "--wheel-dir", build_dir,
    "lifelines",
    "autograd-gamma>=0.3",
    "formulaic>=0.2.2",
    "interface-meta>=1.2.0"
])

print(f"Wheel files created in {build_dir} directory")