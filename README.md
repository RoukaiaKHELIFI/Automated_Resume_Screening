# Automated Resume Screening

Welcome to the Automated Resume Screening project! This repository contains the implementation of a novel framework for automated resume screening using Large Language Models (LLM) agents. The primary goal of this project is to enhance the efficiency and accuracy of the resume screening process, thereby reducing labor costs and saving time.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)

## Overview
The automation of resume screening is a crucial aspect of the recruitment process, especially for large corporations that receive numerous applications. This project leverages advanced NLP techniques and LLMs to build an automated resume screening system that can classify, grade, summarize, and make decisions on resumes.

### Key Objectives
- Efficiently screen a large volume of resumes.
- Automatically classify sentences in resumes.
- Grade and summarize resumes.
- Make informed decisions about candidate selection.

## Features
- **Sentence Classification**: Uses LLaMA2 model for classifying resume sentences.
- **Grading & Summarization**: Employs GPT-3.5 Turbo model to grade and summarize resumes.
- **Decision Making**: Utilizes LLM agents to select qualified candidates.
- **Efficient Processing**: Achieves significant time reduction compared to manual screening.

## Project Structure
```plaintext
Automated_Resume_Screening/
│
├── data/
│   ├── raw/              # Raw resume files
│   ├── processed/        # Processed and cleaned data
│   └── datasets/         # Final datasets for modeling
│
├── models/
│   ├── sentence_classification/  # Scripts and models for sentence classification
│   ├── grading_summarization/    # Scripts and models for grading and summarization
│   └── decision_making/          # Scripts and models for decision making
│
├── notebooks/
│   ├── data_preparation.ipynb    # Jupyter notebook for data preparation
│   ├── modeling.ipynb            # Jupyter notebook for building and training models
│   └── evaluation.ipynb          # Jupyter notebook for evaluating models
│
├── scripts/
│   ├── preprocess.py             # Script for data preprocessing
│   ├── train_model.py            # Script for training models
│   └── evaluate_model.py         # Script for evaluating models
│
├── README.md
└── requirements.txt
