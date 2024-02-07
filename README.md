#  Introduction

The code implements a simple web-based experiment manager using the
 FastAPI framework. This manager is designed to handle machine learning
 experiments, specifically training and evaluating models. The system supports
 asynchronous job execution, progress tracking, and model access.

## Preparation

```bash
pip install-r requirements.txt
```

## Start the app

```bash
 python app.py
```

## App features

### 1. Experiment Management
Experiments are uniquely identified by an experiment ID **(exp_id)**, generated based on
 the current timestamp. The system supports both training jobs and hyperparameter
 search jobs. Progress information is tracked and can be retrieved through the
 **/progress** endpoint.
### 2. Model Evaluation and Access
 Trained models can be evaluated using the **/evaluate** endpoint, and the resulting
 information, along with experiment parameters, is returned. Additionally, users can
 access the trained models using the **/download_model** endpoint.
### 3. Batch jobs
The app can do grid search by **batch size, learning rate or epoch** to find the best hyperparameter there is for the model.
