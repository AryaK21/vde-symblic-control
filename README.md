# VDE Symbolic Control: Physics-Informed Plasma Stabilization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-purple.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official simulation and model-discovery codebase for the paper: **"Simulating Nuclear Plasma Movement Detection Using Physics-Informed Symbolic Regression: A Comprehensive Framework for Magnetohydrodynamic Stability."**

This project introduces a "grey-box" control architecture that bridges high-dimensional deep reinforcement learning (DRL) with classical analytical physics. By training a Proximal Policy Optimization (PPO) agent within a 1D kinematic digital twin and subsequently distilling its policy using Physics-Informed Symbolic Regression (PiSR), we autonomously discover a highly interpretable Proportional-Derivative (PD) control law capable of stabilizing Vertical Displacement Events (VDEs) in tokamak reactors.

---

## 🚀 Key Results

* **Automated Discovery:** The AI autonomously derived a classical control law from operational data: $u(t) = -0.0915z + 0.2161\dot{z} + 0.2374$.
* **Computational Efficiency:** The distilled algebraic controller executes **5,495x faster** natively than the deep learning oracle.
* **Sub-Microsecond Latency:** Forward-pass inference time reduced from 683.875 $\mu$s (PPO Neural Network) to **0.124 $\mu$s** (PiSR Algebraic Law).

---

## ⚡ Quickstart: Google Colab (Recommended)

Because symbolic regression relies on compiling a high-performance Julia backend, the fastest way to reproduce these results without configuring a local environment is via Google Colab. 

1. Upload the project scripts to a new Google Colab notebook.
2. Run the following cell to automatically configure the Julia + Python environment:
   ```python
   !wget [https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz](https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz)
   !tar -xvzf julia-1.10.0-linux-x86_64.tar.gz
   !sudo cp -r julia-1.10.0/* /usr/local/
   !pip install -q pysr stable-baselines3 "shimmy>=2.0"
   import pysr
   pysr.install()
   ```
3. Execute the pipeline scripts sequentially.

---

## ⚙️ Local Installation

If you prefer to run the codebase on bare-metal hardware, you will need to install both Python (3.10+) and Julia (1.10+).

### 1. Install System Dependencies (Julia)

**Linux (Arch):**
```bash
sudo pacman -S julia
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt update
sudo apt install julia
```

**Windows:**
1. Download the installer from the [official Julia website](https://julialang.org/downloads/).
2. Run the executable and ensure you check the box to **"Add Julia to PATH"**.

### 2. Clone the Repository
```bash
git clone [https://github.com/AryaK21/vde-symbolic-control.git](https://github.com/AryaK21/vde-symbolic-control.git)
cd vde-symbolic-control
```

### 3. Python Environment Setup
It is highly recommended to isolate the project dependencies using a virtual environment.

**Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Initialize PySR
Run the internal PySR setup to compile the Julia core and link it with your Python environment:
```bash
python -c "import pysr; pysr.install()"
```

---

## 💻 Usage Pipeline

Once the environment is configured, execute the modules in order:

**Step 1: Train the DRL Oracle**
Trains the PPO neural network to stabilize the plasma and harvests 10,000 operational trajectories to disk.
```bash
python src/train_oracle.py
```

**Step 2: Execute Equation Discovery**
Ingests the trajectory data and utilizes the PySR backend to discover the sparse algebraic control law.
```bash
python src/run_pysr.py
```

**Step 3: Benchmark and Visualize**
Runs native microsecond latency tests comparing the neural network to the algebraic equation, and generates the dual-axis and phase-space `.pdf` publication plots.
```bash
python src/benchmark_plots.py
```

---

## 📄 Citation

If you utilize this codebase, environment, or methodology in your own research, please cite our paper:

```bibtex
@article{kukkadwal2026greybox,
  title={Simulating Nuclear Plasma Movement Detection Using Physics-Informed Symbolic Regression: A Comprehensive Framework for Magnetohydrodynamic Stability},
  author={Kukkadwal, Arya and Oghe, Priya},
  journal={Springer Nature},
  year={2026}
}
```

---

## 👨‍💻 Author
**Arya Kukkadwal**
* Lead Presenter & Researcher 
* Department of Information Technology, Pimpri Chinchwad College of Engineering & Research
