# Explainable Deep Learning using Grad-CAM for the Detection of Intra-Abdominal Bleeding in Trauma CT Scans

This repository contains the source code for the MSc Data Science thesis project, which develops and validates a weakly supervised, multi-task deep learning model for patient-level detection of active abdominal bleeding.

## Project Overview

The core of this project is a hybrid 2.5D CNN-RNN architecture that uses only patient-level labels to predict the presence of hemorrhage. An auxiliary organ segmentation task is used as an anatomical regularizer to improve performance and interpretability. The model was developed using PyTorch and evaluated on the RSNA 2023 Abdominal Trauma Detection dataset.

## Dataset

This model was trained on a curated subset of the **RSNA 2023 Abdominal Trauma Detection dataset**. Due to its large size (~1 TB) and licensing, the dataset is not included in this repository.

To run this code, you must first download the dataset from the official Kaggle competition page:
[https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)

You will also need the `totalsegmentator` organ segmentation masks provided within the competition data.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [[INSERT YOUR GITHUB REPOSITORY URL HERE](https://github.com/ramsjosh/MSc_Proj_Abdominal_Bleeding_AI)]
    cd MSc-Thesis-Abdominal-Bleeding-AI
    ```

2.  **Create a Python environment:** It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:** All required libraries are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is structured as a sequence of Jupyter Notebooks (`.ipynb`) that should be run in order. The main notebooks in the repository are cleaned of their outputs for clarity.

Before running, ensure you have configured the data paths in the first cell of each notebook to point to your local dataset directories.

1.  **`00_create_subset.ipynb`**: Creates the balanced 430-patient subset used in the thesis from the full downloaded dataset.
2.  **`01_preprocess.ipynb`**: Processes the raw DICOM files, performs triple windowing, and creates the 2.5D stacks.
3.  **`02_model_train.ipynb`**: Trains the CNN-RNN model using the preprocessed data.
4.  **`03_evaluate.ipynb`**: Evaluates the trained model on the test set and generates the final performance metrics and Grad-CAM visualizations.

## Static Notebooks with Outputs

For a static, view-only version of the notebooks with all outputs and generated figures included, please see the exported PDF files located in the `/notebook_outputs` folder of this repository.
