# HPC with Python

# Overview

Focuses on developing Python programs for AI applications that utilize HPC resources. It covers basic data processing, normalization, vector/matrix operations, and performance comparisons using Python, NumPy, and PyTorch.

## Project Structure

```
Assignment 2/
├── venv/                       # Python virtual environment
├── Assignment2_Template_files/ # All assignment scripts and data
│   ├── data_loader.py
│   ├── data_process1.py
│   ├── data_process2.py
│   ├── data_tester.py
│   └── vector_product.py
├── GasProperties.csv           # Dataset now at repository root
├── Assignment2_HPC_Python.pdf  # Assignment instructions
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Create and activate a virtual environment:**

   ```
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # Or on Command Prompt:
   venv\Scripts\activate.bat
   ```
2. **Install required packages:**

   ```
   pip install numpy pandas torch typeguard
   ```
3. **Run and test your scripts as needed.**

---

## Matrix Multiplication Precision Comparison

To compare matrix multiplication using 32-bit and 64-bit precision:

- Run `data_tester.py` to perform matrix multiplication on your feature matrix (`X_data`) using both 32-bit and 64-bit precision.
- The script will print computation times for both and explain any observed difference.
- Typical result: 64-bit (float64) operations are slower than 32-bit (float32) due to higher memory and computational requirements. Most consumer hardware is optimized for 32-bit operations.

---

---

### If you rename the `venv` folder

If you change the name of your virtual environment folder (e.g., from `venv` to `myenv`), update all commands and references to use the new folder name:

```
# On Windows PowerShell:
.\myenv\Scripts\Activate.ps1
# Or on Command Prompt:
myenv\Scripts\activate.bat
```

If you use VS Code and have a `.vscode/settings.json` file with a Python interpreter path, update it to point to the new venv folder.

## Main Files

- `data_loader.py`: Central module for all data processing functions.
- `data_process1.py`, `data_process2.py`: Where you implement and test functions before migrating to `data_loader.py`.
- `vector_product.py`: Functions for dot product and matrix operations.
- `data_tester.py`: Script to test your functions.
- `GasProperties.csv`: Dataset for processing (moved to project root). Update any scripts in `Assignment2_Template_files/` to use a path like `../GasProperties.csv` or use the robust path builder in `data_tester.py`.

## Notes

- Do not modify function type hints; they are required for autograder compatibility.
- Use only pure Python (no NumPy) in `normalize_array`, but use NumPy in `normalize_array_np`.
- See the assignment PDF for detailed requirements and instructions.

## Author

Toan Tran [ttran8276@csu.fullerton.edu](mailto:ttran8276@csu.fullerton.edu)

Zero Tsang [ztsang@csu.fullerton.edu](mailto:ztsang@csu.fullerton.edu)

Joseph Rangel [josephrangel@csu.fullerton.edu](mailto:josephrangel@csu.fullerton.edu)
