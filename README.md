
# Project Title: MRI Data Classifier with Independent Evaluation

## Overview
This project involves enhancing an existing Python-based MRI data classifier for more efficient symbol and text detection. The key focus was on integrating an independent evaluation phase, enabling the classifier to be trained and then evaluated on a test dataset seamlessly.

## Table of Contents
- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Environment Setup](#environment-setup)
- [Initial Issues and Modifications](#initial-issues-and-modifications)
- [Key Steps Undertaken](#key-steps-undertaken)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
- [Results](#results)

## Data Preparation
Ensure your data is correctly set up for training and testing:

**Add the train data named as adni and test data as adni2 in the main repo** (if not already present):


## Environment Setup
To create and activate a new Python virtual environment:
```bash
python3 -m venv adn_env
source adn_env/bin/activate
```

To install required libraries:
```bash
pip install torch torchvision numpy pandas pyyaml scikit-learn matplotlib lightly
```

## Initial Issues and Modifications
### Challenges Encountered
1. **Module Import Errors**: Faced `ModuleNotFoundError` for `torch`, `yaml`, `sklearn`, and `lightly`.
2. **NumPy Compatibility Issue**: Encountered an error due to incompatibility between NumPy 2.x and modules compiled with NumPy 1.x.
3. **TorchMetrics Initialization**: The `Accuracy` class failed due to missing parameters, requiring updates to include the `task='multiclass'` argument.
4. **Data Loader Tuple Unpacking Error**: The data loader returned more items than expected, leading to an unpacking error.
5. **Independent Evaluation Configuration**: No dataset was initially specified for independent evaluation.

### Key Fixes
- Downgraded to a compatible version of NumPy (`<2.0`).
- Updated TorchMetrics instantiation to use `task='multiclass'` for correct initialization.
- Adjusted the training loop for correct tuple unpacking.
- Modified the code to handle loading .nii MRI files with nibabel.
- Added and configured a test dataset for independent evaluation.

## Key Steps Undertaken
### 1. Understanding and Resolving Import Issues
- Installed missing modules (`torch`, `yaml`, `scikit-learn`, `lightly`).
- Addressed NumPy compatibility issues by downgrading to version `<2.0`.

### 2. Code Modifications
- Corrected instantiation errors in `torchmetrics` classes by adding required parameters like `task='multiclass'`.
- Resolved data loader unpacking by ensuring that the training loop handled three items (inputs and labels) returned per iteration.
- Added `if __name__ == "__main__":` to main scripts to handle multiprocessing safely.

### 3. Configuration Adjustments
- Ensured all paths in `configuration.yaml` were correctly set for data and checkpoint files.
- Added a test dataset to the `independent_evaluation` section of the configuration file.

### 4. Training and Independent Evaluation Implementation
- Updated the training code to properly log performance metrics such as accuracy, recall, MCC, and specificity.
- Enhanced the independent evaluation script (`run_classifier_indep_eval.py`) to evaluate the trained model and log the test metrics.

### 5. Final Testing and Logging Enhancements
- Verified that the training and evaluation phases worked seamlessly, with comprehensive logging for easier troubleshooting.
- Created a detailed README-style documentation for ease of project understanding and replication.

## Installation and Dependencies
Below is the content of `requirements.txt` generated for this project:
```plaintext
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
pandas==2.2.3
pyyaml==6.0.2
scikit-learn==1.14.1
matplotlib==3.9.2
lightly==2.3.0
```

## Usage
### Running the Training Script
```bash
python run_train_nnclr.py
```

### Running the Independent Evaluation Script
```bash
python run_classifier_indep_eval.py
```

## Results
The final test results included performance metrics such as confusion matrix, Matthews correlation coefficient (MCC), and recall. Further modifications can be made to optimize model accuracy and balance.

---

This README provides a comprehensive guide for understanding, running, and modifying the MRI data classifier project. For further questions or contributions, feel free to reach out or submit a pull request.
