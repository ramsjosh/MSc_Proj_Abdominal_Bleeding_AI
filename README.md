# MSc_Proj_Abdominal_Bleeding_AI
An explainable deep learning model using Grad-CAM for detecting intra-abdominal bleeding in trauma CT scans.

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
    git clone [https://github.com/ramsjosh/MSc_Proj_Abdominal_Bleeding_AI]
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

The project is structured into a sequence of scripts that should be run in order.

**0. Dataset Curation (Optional but Recommended)**

This script creates the balanced 430-patient subset used in the thesis from the full downloaded dataset.

```bash
python 00_create_subset.py
```

**1. Data Preprocessing:**
This script processes the raw DICOM files from your curated subset, performs triple windowing, and creates the 2.5D stacks.
```bash
python 01_preprocess.py --data_dir /path/to/bleed_subset_images --output_dir /path/to/preprocessed_data
```

**2. Model Training:**
This script trains the CNN-RNN model using the preprocessed data.
```bash
python 02_train.py --preprocessed_dir /path/to/preprocessed_data --save_path /path/to/save/model.pth
```

**3. Evaluation:**
This script evaluates the trained model on the test set and generates the final performance metrics and Grad-CAM visualizations.
```bash
python 03_evaluate.py --model_path /path/to/save/model.pth --preprocessed_dir /path/to/preprocessed_data

