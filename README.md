# ECG Arrhythmia Classification with Machine Learning

## Overview

This project implements a comprehensive machine learning pipeline for automatic classification of electrocardiogram (ECG) signals into different arrhythmia types based on the AAMI (Association for the Advancement of Medical Instrumentation) standards. The system combines multiple feature extraction techniques and ensemble learning methods to achieve high classification accuracy.

## Key Features

- **Advanced Feature Extraction**: Wavelet transforms, Higher Order Statistics (HOS), RR intervals, and morphological features
- **Multiple Classifiers**: Support Vector Machines (SVM), Random Forest, LightGBM, and ensemble methods
- **Class Imbalance Handling**: SMOTE, ADASYN, and class weighting techniques
- **Comprehensive Evaluation**: AAMI-standard performance metrics and visualizations
- **Web Interface**: Streamlit-based web application for easy interaction

## Performance Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 86.66% | 0.96 | 0.87 | 0.91 |
| LightGBM | 81.49% | 0.96 | 0.81 | 0.88 |
| Advanced Ensemble | 89.89% | 0.96 | 0.90 | 0.93 |

<img width="578" height="418" alt="image" src="https://github.com/user-attachments/assets/f9e0f083-02e9-4641-ba4f-b8a31fc0876b" />
<img width="649" height="500" alt="image" src="https://github.com/user-attachments/assets/4f89da0c-6a25-4935-86e4-0654e79db300" />
<img width="635" height="473" alt="image" src="https://github.com/user-attachments/assets/aa0a7398-22e3-4f1c-bba1-8c25216ddb05" />




## AAMI Classification Standards

The system classifies beats into five superclasses according to AAMI recommendations:

| SuperClass | Included Beat Types | Description |
|------------|---------------------|-------------|
| **N** (Normal) | N, L, R | Normal beats and bundle branch blocks |
| **SVEB** (Supraventricular) | A, a, J, S, e, j | Supraventricular ectopic beats |
| **VEB** (Ventricular) | V, E | Ventricular ectopic beats |
| **F** (Fusion) | F | Fusion beats |
| **Q** (Unknown) | /, f, Q | Unknown and unclassifiable beats |

##  Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```txt
streamlit==1.36.0
numpy==1.26.4
scipy==1.13.1
pywavelets==1.6.0
biosppy==0.7.2
scikit-learn==1.5.0
joblib==1.4.2
matplotlib==3.8.4
pandas==2.2.2
lightgbm==4.1.0
wfdb==4.3.0
imbalanced-learn==0.12.0
tqdm==4.66.0
```

## Quick Start

### 1. Download the Dataset

The MIT-BIH Arrhythmia Database can be obtained from:

- **Kaggle**: [MIT-BIH Arrhythmia Database](https://www.kaggle.com/mondejar/mitbih-database)
- **PhysioNet**: Using WFDB tools:
  ```bash
  rsync -Cavz physionet.org::mitdb /path/to/save/mitdb
  ```

### 2. Run the Training Pipeline

```python
# Initialize data loader
data_loader = ECGDataLoader("/path/to/mit-bih-arrhythmia-database")

# Load specific patients
train_patients = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119']
test_patients = ['100', '103', '105', '111', '113']

train_signals, train_annotations, train_rpeaks = data_loader.load_mit_bih_data(train_patients)
test_signals, test_annotations, test_rpeaks = data_loader.load_mit_bih_data(test_patients)

# Initialize classifier and extract features
classifier = ECGClassifier(sampling_rate=360)
X_train, y_train = classifier.extract_features_from_dataset(train_signals, train_annotations, train_rpeaks)
X_test, y_test = classifier.extract_features_from_dataset(test_signals, test_annotations, test_rpeaks)

# Train and evaluate
model, feature_selector = classifier.train_lightgbm_model_improved(X_train, y_train, "improved_model")
predictions = classifier.predict(X_test, "improved_model")
```

### 3. Launch the Web Application

```bash
streamlit run ecg_app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
ecg-classification/
├── ecg_app.py              # Streamlit web application
├── requirements.txt        # Python dependencies
├── models/                 # Trained model files
│   └── ecg_classifier_advanced_pipeline.pkl
├── utils/
│   ├── data_loader.py     # ECG data loading and preprocessing
│   ├── feature_extractor.py # Feature extraction methods
│   └── classifier.py      # Machine learning models
└── notebooks/             # Jupyter notebooks for experimentation
```

## Feature Extraction

The system extracts four types of features from each ECG beat:

### 1. Morphological Features (180 features)
- Raw signal values from a window of [-90, 90] samples around R-peak
- Downsampled to 90 features

### 2. Wavelet Features (10 features)
- Daubechies 1 (db1) wavelet with 3 levels of decomposition
- Top 10 approximation coefficients

### 3. Higher Order Statistics (10 features)
- Skewness and kurtosis calculated over 6 intervals
- Captures non-linear signal characteristics

### 4. RR Interval Features (8 features)
- Pre-RR, post-RR, local-RR, and global-RR intervals
- Both raw and normalized versions

**Total: 118 features per beat**

## Machine Learning Approach

### Data Preprocessing
- Baseline wander removal using median filters
- R-peak detection using BioSPPy
- Beat segmentation around R-peaks
- Z-score normalization of features

### Class Imbalance Handling
- **Class weighting**: Adjusting class weights in loss functions
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **Threshold optimization**: Class-specific prediction thresholds

### Model Architecture
- **LightGBM**: Gradient boosting with class weighting
- **Random Forest**: Ensemble of decision trees with balanced subsampling
- **Voting Classifier**: Ensemble of multiple models
- **Feature Selection**: SelectKBest with ANOVA F-value

## Evaluation Metrics

The system uses AAMI-standard evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value for each class
- **Recall**: Sensitivity for each class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-wise performance
- **JK Index**: Comprehensive performance measure

## Web Application Features

The Streamlit web app provides:

- **ECG Signal Input**: Upload CSV files or paste raw values
- **Real-time Analysis**: Instant classification results
- **Visualizations**: ECG signal plots and confidence scores
- **Sample Data**: Generate demo ECG signals for testing
- **Detailed Reports**: Class-wise performance metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@article{MONDEJARGUERRA201941,
  title = {Heartbeat classification fusing temporal and morphological information of ECGs via ensemble of classifiers},
  author = {Mondéjar-Guerra, V and Novo, J and Rouco, J and Penedo, M G and Ortega, M},
  journal = {Biomedical Signal Processing and Control},
  volume = {47},
  pages = {41--48},
  year = {2019},
  doi = {https://doi.org/10.1016/j.bspc.2018.08.007}
}
```

## Dataset Information

### MIT-BIH Arrhythmia Database
- **Sampling Rate**: 360 Hz
- **Duration**: 30 minutes per record
- **Leads**: 2 leads (MLII usually preferred)
- **Patients**: 47 subjects
- **Annotations**: Beat-level and rhythm-level annotations

### Inter-patient Split
The dataset is split using the inter-patient scheme to ensure no patient overlap between training and testing:

**Training Set (22 patients)**: 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230

**Test Set (22 patients)**: 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234

## Important Notes

- This is a research implementation and should not be used for clinical decision-making without proper validation
- The model performance may vary with different ECG recording devices and conditions
- Always consult healthcare professionals for medical diagnosis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- MIT-BIH Arrhythmia Database providers
- PhysioNet for maintaining the database
- Contributors to the open-source libraries used in this project

---

**Disclaimer**: This software is intended for research purposes only. It should not be used for medical diagnosis or treatment without consultation with qualified healthcare professionals.
