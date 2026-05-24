# Setup & Hardware Guide

ChurnBot is designed to run fully locally with no external services, API keys, or cloud dependencies.

This guide covers local installation, virtual environment setup, OS-specific activation commands, and practical hardware recommendations for running the ChurnBot research pipeline.

---

## Prerequisites

- **Python 3.14+** recommended
- **Git**
- **Virtual environment tool**: `venv`, `conda`, or equivalent

> `venv` is Python’s built-in virtual environment system and is sufficient for most local setups.  
> Conda environments may also be used as an alternative.

---

## Hardware Requirements

ChurnBot can run on modest hardware, but full research workflows such as EBM training, Optuna sweeps, notebook execution, and diagnostics benefit from additional memory and CPU capacity.

### Lightweight / Minimal Setup

- 8GB RAM  
  *Usable with a lightweight Windows 10/11 setup*
- Intel i5 or equivalent CPU
- SSD strongly recommended
- Minimal background applications/services

Models have been successfully trained and developed on an optimized Windows 10 installation with 8GB RAM using only VS Code with the training pipeline active.

During EBM training, memory usage typically ranges around **~5.5–7GB RAM** depending on workload and multitasking.

However, browser usage and additional background applications can quickly exhaust available memory on 8GB systems, especially during Optuna hyperparameter optimization and extended EBM training sessions.

---

### Recommended for Comfortable Development

- 16GB RAM
- 4+ core CPU  
  *Ryzen 5 / Intel i5 equivalent or better*
- SSD or NVMe SSD

This configuration provides substantially smoother multitasking, improved notebook responsiveness, and more stable performance during longer research sessions and hyperparameter sweeps.

---

### Recommended for Heavy Research & Hyperparameter Optimization

- 32GB+ RAM
- 8+ core CPU  
  *Ryzen 7 / Intel i7 equivalent or better*
- NVMe SSD

Recommended for:

- large Optuna sweeps
- simultaneous notebooks/browser workloads
- extensive EBM tuning
- future GlassCUDA experimentation
- parallel experimentation and diagnostics

---

## Quick Start

Open a terminal and navigate to your local projects/workspace directory.

Example:

```text
C:\Users\User\Projects
```

Install Python 3.14+ from:  
https://www.python.org/downloads/

Then clone the repository and create a virtual environment:

```bash
git clone https://github.com/HKtrill/ai-business-coach.git
cd ai-business-coach
python -m venv .venv
```

> **Important:** Make sure you are inside the `ai-business-coach` root directory when creating the virtual environment.  
> This helps avoid interpreter conflicts and ensures modular package imports resolve correctly.

---

### macOS / Linux

**Activate the virtual environment:**

```bash
source .venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

---

### Windows PowerShell Setup

**Activate the virtual environment:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Install dependencies:**

```powershell
pip install -r requirements.txt
```

---

### 📝 Notes

- All computation runs locally.
- No API keys are required.
- No cloud services are required.
- Supported on Windows, macOS, and Linux.
- VS Code is recommended for development and notebook execution.
