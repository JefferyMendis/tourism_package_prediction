Tourism Package Prediction – MLOps Project

This project implements an end-to-end MLOps workflow designed to predict whether a customer is likely to purchase a Wellness Tourism Package. It brings together data preparation, model training, experiment tracking, model versioning, automated CI/CD, and application deployment into a fully streamlined pipeline.

The goal is to support marketing and sales teams with reliable predictions while showcasing modern MLOps practices that ensure scalability, reproducibility, and continuous improvement of the model lifecycle.

Project Overview

The solution begins with collecting and preparing customer interaction data, followed by a structured machine learning pipeline. The data undergoes cleaning, preprocessing, and transformation before being used to train multiple model configurations. Experimentation is tracked using MLflow, enabling comparison of hyperparameters, metrics, and artifacts over time.

Once the best-performing model is selected, it is packaged, versioned, and stored on the Hugging Face Model Hub. The project then integrates deployment capabilities, allowing real-time predictions through a Streamlit application served via Docker or Hugging Face Spaces.

Key Features
1. Dataset Registration & Management

The dataset is uploaded and maintained in a Hugging Face Dataset Repository. This ensures consistent access to clean, versioned, and centrally stored data, especially useful for distributed training or CI workflows.

2. Data Preparation Workflow

A reproducible preprocessing pipeline handles:

Data cleaning and label standardization

Train/test splitting

Saving processed splits

Uploading the processed data back to the dataset repository

This allows all environments—local, CI, or production—to work with identical data.

3. Model Training & Experiment Tracking

Multiple training strategies are implemented:

A development training pipeline for rapid experimentation

A production-grade pipeline with robust early-stopping mechanisms

MLflow is used for tracking:

Training metrics

Hyperparameters

Comparisons across runs

Model artifacts

This provides transparency and reproducibility across iterations.

4. Model Versioning & Publishing

The final model is registered and stored in the Hugging Face Model Hub. This ensures:

Version control

Easy sharing

Seamless integration with downstream applications

Both development and production models are maintained in separate model repositories.

5. Application Deployment

A Streamlit application is built to serve predictions through an intuitive interface. The app loads the trained model from Hugging Face, collects user inputs, and displays the predicted outcome clearly.

Deployment is supported in two ways:

Docker-based deployment, enabling consistent containerized execution

Hugging Face Spaces, making the app publicly accessible without provisioning servers

6. CI/CD Automation

A GitHub Actions workflow orchestrates the full pipeline:

Register dataset

Prepare data

Train model

Upload and deploy application

This automation ensures every push triggers an updated and fully traceable ML cycle, contributing to reliability and continuous integration.

Dependency & Environment Handling

To maintain reproducibility, all dependencies are centrally managed in a dedicated requirements.txt file. The project avoids hard-coded secrets, instead relying on environment variables and GitHub Secrets within CI pipelines. This safeguards sensitive credentials and enforces secure development practices.

Hosting & Deployment Strategy

The project leverages the Hugging Face Hub for:

Hosting datasets

Managing models

Deploying the interactive prediction application

By centralizing the entire lifecycle on a single platform, the project simplifies model consumption while offering transparency and accessibility.

Conclusion

This Tourism Package Prediction project demonstrates the complete lifecycle of an ML model—from raw data to deployment—implemented with modern, production-oriented MLOps practices. By combining experiment tracking, automated pipelines, cloud hosting, and an accessible user interface, the system is structured to support continuous improvement and easy scalability.

If you're exploring how to operationalize machine learning or build reliable predictive systems, this project serves as a comprehensive and practical reference.
