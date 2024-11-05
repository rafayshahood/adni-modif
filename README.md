
# MRI Data Classifier with Independent Evaluation

## Overview
This project is designed for training and evaluating an MRI data classifier. The workflow involves training the model, creating checkpoints, and running an independent evaluation.

## Table of Contents
- [Overview](#overview)
- [Workflow](#workflow)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Verification](#verification)

## Workflow
1. **Prepare Datasets**:
   - Place the training dataset in a folder named `adni` in the main repository directory.
   - Place the testing dataset in a folder named `adni2` in the main repository directory.

2. **Set Up Virtual Environment and Install Libraries**:
   - Create a virtual environment and activate it:
     ```bash
     python3 -m venv adn_env
     source adn_env/bin/activate
     ```
   - Install all necessary libraries using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Training for NNCLR**:
   - Train the NNCLR model using:
     ```bash
     python run_train_nnclr.py
     ```
   - This will generate a new checkpoint file in the `output/checkpoints` directory. **Verify that the checkpoint file is created successfully.**

4. **Run the Classifier Training**:
   - Train the classifier by running:
     ```bash
     python run_train-classifier.py
     ```
   - This will create a new checkpoint file in the `output/checkpoints` directory. **Ensure that the checkpoint file is created as expected.**

5. **Evaluate the Model**:
   - Run the independent evaluation script to evaluate the model on the test dataset:
     ```bash
     python run_classifier_indep_eval.py
     ```

## Environment Setup
Ensure that all dependencies are installed from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Training Scripts
1. **NNCLR Training**:
   ```bash
   python run_train_nnclr.py
   ```

2. **Classifier Training**:
   ```bash
   python run_train-classifier.py
   ```

### Running the Evaluation Script
Evaluate the model using:
```bash
python run_classifier_indep_eval.py
```

## Verification
- Check that the output files in `output/checkpoints` are generated after each training phase.
- Review logs to confirm that the scripts ran without errors.

## Notes
- Ensure the working directory is set properly in `configuration.yaml` and paths are relative for easy portability.
- The datasets (`adni` and `adni2`) should be in place before running any training or evaluation scripts.

This structured workflow will guide you from setting up the environment, running the training scripts, to evaluating the model effectively.
