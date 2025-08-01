MLOps Assignment - California Housing Linear Regression Pipeline
This repository contains a complete MLOps pipeline for Linear Regression using the California Housing dataset from sklearn. The pipeline includes training, testing, quantization, Dockerization, and CI/CD automation.

<img width="213" height="517" alt="image" src="https://github.com/user-attachments/assets/c77c50a0-82cc-4bef-82e2-bfff43d3580a" />


Set Up Project Structure

Created folders: src/, tests/, .github/workflows/

Added scripts: train.py, predict.py, quantize.py, utils.py

Data Loading & Model Training

Loaded the California Housing dataset

Trained a Linear Regression model using train.py

Saved the model using joblib

Model Quantization

Quantized and saved model parameters via quantize.py

Model Prediction

Loaded the trained model

Performed inference and printed sample predictions using predict.py

Testing with Pytest

Created tests/test_train.py to test model training output and performance

Set up PYTHONPATH to enable relative imports from src/

CI/CD Pipeline with GitHub Actions

Defined .github/workflows/ci.yml

Configured pipeline to:

Set up Python

Install dependencies

Run tests

Train and quantize model

Upload artifacts

Build and test Docker image

Dockerization

Wrote a Dockerfile to containerize the app

Verified model runs correctly inside the Docker container
