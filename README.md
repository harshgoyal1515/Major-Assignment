# MLOps Regression Pipeline- California Housing Linear Regression Pipeline

This repository demonstrates an end-to-end MLOps workflow for a Linear Regression model using the California Housing dataset. It includes training, model quantization, automated testing with GitHub Actions, and Docker containerization.
# Project Structure
 <img width="266" height="646" alt="image" src="https://github.com/user-attachments/assets/5bd7de52-3bcb-47e3-82c6-b0e7984f2145" />

Step 1: Create Virtual Environment
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Step 2: Train the Model
python src/train.py
<img width="940" height="226" alt="image" src="https://github.com/user-attachments/assets/1b5850e8-46ea-4c2e-96a1-d499a9fb9d2f" />

This will:
•	Load the California Housing dataset
•	Train a Linear Regression model
•	Print R² score and loss metrics
•	Save the trained model as model.joblib
    

Step 3: Quantize the Model
python src/quantize.py

This will:
Load the trained model
Extract coefficients and intercept
Save raw parameters as unquant_params.joblib
Quantize parameters to 8-bit unsigned integers
Save quantized parameters as quant_params.joblib
Perform inference with de-quantized weights
 <img width="940" height="148" alt="image" src="https://github.com/user-attachments/assets/8a0e53dd-6d22-436f-92ac-c586f15af74f" />


Step 4: Predict
python src/predict.py

 <img width="940" height="122" alt="image" src="https://github.com/user-attachments/assets/963b620e-85f1-4aaa-997f-bb9353bb53f9" />

Step 5: Run Tests
export PYTHONPATH=src
pytest tests/

CI/CD with GitHub Actions
The .github/workflows/ci.yml file runs:
•	Tests on every push to main
•	Model training and quantization
•	Docker build and test

# Build Docker image
docker build -t mlops-regression .

# Run Docker container
docker run mlops-regression

# DockerHub Image
You can pull the latest Docker image from DockerHub: https://hub.docker.com/repository/docker/harshgoyal15/major/tags
docker pull harshgoyal15/major
docker run --rm harshgoyal15/major

Artifacts
GitHub Actions will upload the following:
•	model.joblib
•	quant_params.joblib
•	unquant_params.joblib
These are downloaded later for Docker testing.
Metric	Value
Model Type	Linear Regression
Dataset	California Housing (sklearn)
R² Score	~0.60 (typical)
Features	8 numerical features
Target	Median house value

Model Performance
Metric	Value
Model Type	Linear Regression
Dataset	California Housing (sklearn)
R² Score	~0.60 (typical)
Features	8 numerical features
Target	Median house value

Quantization Details
•	Precision: 8-bit unsigned integers (0-255)
•	Parameters Quantized: Model coefficients and intercept
•	Format: Manual quantization with scale and zero-point
•	Storage: Separate files for original and quantized parameters
Docker Configuration
The Docker container:
•	Uses Python 3.9 slim base image
•	Installs all required dependencies
•	Includes the trained model and prediction script
•	Runs predict.py on container start
Testing Strategy
Unit tests cover:
•	Dataset loading functionality
•	Model creation and validation
•	Training process verification
•	R² score threshold validation
•	Parameter extraction for quantization
License
This project is for educational purposes as part of an MLOps assignment.

