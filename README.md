# Explainable Deep Learning using Grad-CAM for the Detection of Intra-Abdominal Bleeding in Trauma CT Scans

This repository contains the source code for the MSc Data Science Final Project, which develops and evaluates a weakly supervised, multi-task deep learning model for patient-level detection of intra-abdominal hemorrhage from trauma CT scans.

## Project Overview

The core of this project is a hybrid 2.5D CNN-RNN architecture trained using only patient-level labels to predict the presence of active bleeding. An auxiliary organ segmentation branch is included to act as an anatomical regularizer, improving both performance and interpretability. The model was implemented in PyTorch and evaluated using the RSNA 2023 Abdominal Trauma Detection dataset.

Model explainability was explored using Grad-CAM, enabling visualization of the regions influencing model decisions.

## Dataset

This model was trained on a balanced subset of the **RSNA 2023 Abdominal Trauma Detection dataset**. Due to its large size (~1 TB) and licensing, the dataset is not included in this repository.

To run this code, you must first download the dataset from the official Kaggle competition page:
[https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection)

You will also need the `TotalSegmentator` organ segmentation masks provided within the competition data.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ramsjosh/MSc_Proj_Abdominal_Bleeding_AI.git MSc-Project-Abdominal-Bleeding-AI
    cd MSc-Project-Abdominal-Bleeding-AI
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

The project is organized as a sequence of Jupyter notebooks (.ipynb) that should be executed in order. These are intentionally cleared of outputs for readability.

Before running, make sure to configure the dataset paths in the first cell of each notebook.

1.  **`00_create_subset.ipynb`**: Creates the balanced 430-patient subset used in the MSc Project from the full downloaded dataset.
2.  **`01_preprocess.ipynb`**: Processes the raw DICOM files, performs triple windowing, and creates the 2.5D stacks.
3.  **`02_model_train.ipynb`**: Trains the CNN-RNN model using the preprocessed data.
4.  **`03_evaluate.ipynb`**: Evaluates the trained model on the test set and generates the final performance metrics and Grad-CAM visualizations.

## Static Notebooks with Outputs

To explore the full notebooks including all outputs, visualizations, and evaluation results, you can view them directly on Google Colab (read-only):

[View Static Notebooks with Outputs on Google Colab](https://colab.research.google.com/drive/187ypbuKzcxtKfNXtCBh3KXxMkcYM81A6?usp=sharing)

These include:
-  **Training and Validation Curves** (e.g., AUC, loss, accuracy per epoch)
-  **Confusion Matrix**  to visualize classification breakdown of bleeding vs no bleeding
-  **Dice Score Evaluation**  for segmentation performance on organ masks
-  **ROC and Precision-Recall (PR) Curves** for model evaluation
-  **Grad-CAM Visualizations** showing model attention on bleeding regions
-  **Dataset Summary Statistics** (class balance, scan counts, slice/masks distribution)
-  **Examples of Preprocessing Outputs** (triple windowing, 2.5D stack samples, augmentation samples)
