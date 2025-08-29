# ECG-Based_Health_Risk_Stratification_Platform
ECG-Based Multi-Task Deep Learning Framework for Comprehensive Health Risk Stratification

Architected a multi-task learning (MTL) framework with a shared 1D ResNet-34 backbone and five task-specific heads to predict a spectrum of conditions—from arrhythmias and myocardial infarction to ejection fraction and future mortality risk—from raw ECG signals.

Engineered a complex data unification pipeline to integrate six public ECG datasets (MIT-BIH, PTB-XL, SCDH), resolving heterogeneities in lead configurations, sampling rates, and labeling schemas to create a unified ontology for 15+ cardiac and systemic conditions.

Implemented model explainability using Gradient-weighted Class Activation Mapping (Grad-CAM) to generate visual saliency maps, identifying clinically relevant features within the ECG waveform (e.g., ST-segment for ischemia) and validating decisions against cardiological principles.

Achieved state-of-the-art performance, predicting left ventricular systolic dysfunction (LVSD) from sinus rhythm with an AUC of 0.89 and identifying atrial fibrillation with an F1-score of 0.95, demonstrating the ability to extract sub-visual biomarkers.

Deployed the system as an interactive web application using Gradio, providing a comprehensive health report to facilitate early intervention and showcasing the potential of AI as a scalable tool for preventive cardiology
