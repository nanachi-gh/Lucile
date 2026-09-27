Install Lucile with the following command:
```
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ Lucile-bigshaq9999
```

To run on amd gpu:

```
# 1. Create a Python 3.12 virtual environment
uv venv --python 3.12

# 2. Install PyTorch with ROCm 6.2 support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# 3. Install project dependencies
uv pip install -e .

# 4. Launch Lucile (use --no-sync so uv does not overwrite ROCm torch with PyPI wheels)
uv run --no-sync lucile
```

> **Note on AMD GPUs**: On Linux systems with AMD RDNA3 / Phoenix / Hawk Point APUs (Radeon 780M / gfx1103), `HSA_OVERRIDE_GFX_VERSION=11.0.0` and PyTorch attention compatibility modes are now configured automatically at launch. If any model operation exceeds GPU VRAM, it will gracefully fall back to CPU without halting the pipeline.