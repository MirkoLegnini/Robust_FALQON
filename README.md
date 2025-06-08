# Robust\_FALQON

Pennylane implementation of the FALQON algorithm under coherent control errors and experiments on Robust FALQON.

## Overview

This repository accompanies the paper **Robust Feedback Based Quantum Optimization: analysis of coherent control errors**. It provides:

* An implementation of the FALQON variational quantum algorithm (FALQON.py).
* Problem definitions and utility functions (problem.py).
* Three simulation scripts to reproduce the experiments:

  * `run_systematic.py`: Run comparision for different systematic CCE levels over a single run of FALQON.
  * `run_average_independent.py`: Run comparision for different independent CCE levels over a single run of FALQON, plot the variance of the error on each layer.
  * `compare_epsilon_independent.py`: Compare the error on the final layer for multiple values of epsilon.
  
## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/MirkoLegnini/Robust_FALQON.git
   cd Robust_FALQON
   ```

2. **Create & activate a virtual environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage
Each of the simulation scripts can be run directly from the command line

# Credits
The implementation for FALQON is partially based on the Pennylane tutorial by David Wakeham and Jack Ceroni (https://pennylane.ai/qml/demos/tutorial_falqon). 
