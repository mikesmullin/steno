# Steno CLI Installation Guide

This guide explains how to install Steno as a global CLI command that works with your Python environment.

## Installation Methods

### Method 1: Development Install (Recommended for development)

This creates a symbolic link, so changes to the code are immediately reflected:

```bash
# 1. Activate your Python environment (if using one)
# With venv:
venv\Scripts\activate
# Or with pyenv, conda, etc.

# 2. Install in editable mode
pip install -e .

# 3. Now you can run 'steno' from anywhere!
steno -m -s -o transcript.jsonl
```

**Quick install on Windows:**
```bash
install.bat
```

### Method 2: Regular Install

This installs the package like any other Python package:

```bash
pip install .
```

### Method 3: Using pipx (For global CLI tools)

`pipx` installs CLI tools in isolated environments, making them available globally:

```bash
# Install pipx (if not already installed)
pip install pipx
pipx ensurepath

# Install steno globally
pipx install .

# For development (editable install with pipx)
pipx install -e .

# Now 'steno' works from any directory, any terminal!
steno --help
```

**Benefits of pipx:**
- ✅ Works globally, regardless of active Python environment
- ✅ Each tool in its own isolated environment
- ✅ No conflicts with other packages
- ✅ Easy to uninstall: `pipx uninstall steno-cli`

## Verification

After installation, verify it works:

```bash
# Check if command is available
steno --help

# Check where it's installed
where steno  # Windows
which steno  # Linux/Mac
```

## Uninstallation

```bash
# If installed with pip:
pip uninstall steno-cli

# If installed with pipx:
pipx uninstall steno-cli
```

## Troubleshooting

### Command not found

If `steno` is not found after installation:

1. Make sure your Python Scripts directory is in PATH:
   ```bash
   # Windows - add to PATH:
   %USERPROFILE%\AppData\Local\Programs\Python\Python3XX\Scripts
   # Or for venv:
   <path-to-venv>\Scripts
   ```

2. Restart your terminal

3. Check if it's installed:
   ```bash
   pip show steno-cli
   ```

### Using with pyenv

If using pyenv, make sure to install in the pyenv environment:

```bash
# Select your Python version
pyenv shell 3.11.0

# Install
pip install -e .

# The command will be available in that pyenv environment
steno --help
```

## Development Workflow

For active development:

```bash
# 1. Install in editable mode
pip install -e .

# 2. Make changes to steno.py

# 3. Test immediately (no reinstall needed!)
steno -m -o test.jsonl

# 4. Changes are reflected immediately
```
