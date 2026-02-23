# XAI-Based-Retinal-Disease-Classification-using-Fine-Tuned-ResNet50-with-Grad-CAM

## Overview

This project presents an Explainable AI (XAI) framework for automated multi-class classification of retinal fundus images using a fine-tuned ResNet50 architecture. The system detects the following classes:

- Diabetic Retinopathy (DR)
- Glaucoma
- Cataract
- Normal

The model integrates transfer learning, controlled fine-tuning, and Grad-CAM based visual explanations to enhance interpretability in medical diagnosis.

## Objectives

- Develop a robust deep learning model for retinal disease classification.
- Improve feature specialization through structured fine-tuning.
- Generate visual explanations using Grad-CAM.
- Establish a foundation for explanation consistency and reliability analysis.

## Dataset

The dataset used in this project is publicly available:

Eye Disease Retinal Images Dataset  
https://ieee-dataport.org/documents/four-public-datasets-explainable-medical-image-classifications

Download the dataset and place it in:

eye-disease-dataset/
    train_dir/
    val_dir/
    test_dir/

## Model Architecture

- Backbone: ResNet50 (pretrained on ImageNet)
- Global Average Pooling
- Batch Normalization
- Dense Layer (256 units, ReLU)
- Dropout (0.5)
- Final Dense Layer (Softmax, 4 classes)

### Training Strategy

**Phase 1 – Feature Extraction**
- Frozen ResNet backbone
- Train classification head

**Phase 2 – Fine-Tuning**
- Unfreeze final convolutional layers
- Low learning rate (1e-5)
- Early stopping & learning rate scheduling

## Performance

- Phase 1 Accuracy: ~89%
- Fine-Tuned Model Accuracy: ~92% (validation)

Further class-wise metrics and confusion matrices will be added in future updates.

## Explainability Module

Grad-CAM is applied to:

- Identify salient regions influencing predictions
- Visualize disease-specific attention patterns
- Enable qualitative evaluation of model reasoning

Future work includes:
- Explanation consistency analysis
- Stability under perturbations
- Failure case attention analysis

## Why Explainability?

Medical AI systems must provide interpretability to support clinical trust. This project focuses not only on accuracy but also on understanding model decision behavior.

## Technologies Used

- TensorFlow / Keras
- ResNet50 (Transfer Learning)
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Repository Structure
Step1_Model_Development_ResNet50_FineTuning_GradCAM.ipynb
Future modules will include:
  Step2_Explanation_Storage_and_Analysis.ipynb
  Step3_Consistency_and_Stability_Evaluation.ipynb

## Future Enhancements

- Quantitative explanation similarity metrics
- Class-wise average heatmap analysis
- Attention behavior under input perturbations
- Clinical feature extraction (e.g., optic disc localization)
- Structured diagnosis report generation

## Disclaimer

This project is intended for research and educational purposes only. It is not a clinically validated diagnostic tool.
