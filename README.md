
# DeepFlowState: Demo Repository for *Understanding Deep Learning*

Welcome to the demo repository for the Osnabrück University course *Understanding Deep Learning*, part of the Cognitive Science program.

This repository showcases a practical application of deep learning using **BERT** (Bidirectional Encoder Representations from Transformers), originally introduced by Devlin et al. (2019). It is maintained by the **DeepFlowState** group and is designed to demonstrate our approach to model fine-tuning and transfer learning in the context of pharmacovigilance.

---

## 🧠 Project Overview

The primary goals of this project are:

- **Adverse Drug Reaction (ADR) detection** – classification task  
- **Drug and effect identification** – Named Entity Recognition (NER) task  

ADR detection is a key part of **pharmacovigilance**—the ongoing monitoring of drug safety in real-world usage. Detecting adverse reactions early is crucial, and **NLP methods** applied to **social media posts** and **medical feedback forums** are valuable tools for this task.

We fine-tune and test the pre-trained **BERT** model on the following labeled datasets:

- **ADE Corpus** – General medical case reports  
- **PsyTAR** – Medical feedback forums (psychiatry-focused)  

Each dataset contains **labeled drug-related adverse reactions**.

---

## 🔬 Novelty & Research Focus: Transfer Learning & Generalization

This project examines BERT’s **cross-dataset generalization** via **transfer learning**, moving beyond simple in-domain evaluations.  

Key research questions:

- Can a model fine-tuned on one dataset detect ADRs in a completely different dataset?  
- How well does it **transfer across platforms, annotation guidelines, and drug vocabularies**?  

We evaluate:

- **In-domain**: Model trained and tested on the same dataset  
- **Out-of-domain**: Model trained on one dataset and tested on the other  

This approach helps us assess whether the model captures the **core ADR concept** beyond dataset-specific patterns.

---

## 📁 Repository Contents

This repository includes:
trainer/ # Scripts for fine-tuning BERT
├─ train_ner.py # 3-fold CV for NER
└─ train_classification.py # 3-fold CV for classification

evaluation/ # Evaluation scripts
├─ evaluate_classification.py
└─ evaluate_ner.py

data_sets/ # Datasets for training and testing

jupyter_notebooks/ # Visualization & results
└─ ade_param_comparison.ipynb

utils/
preprocessing/
model_selection/

requirements.txt # Runtime dependencies
requirements-dev.txt # Dev dependencies
.gitignore
README.md # This file

---

## 🚀 Quickstart

Follow these steps to get the project up and running locally.

### 1. Clone the Repository

```bash
git clone https://github.com/deepflowstate/ADE_Classifier-demo.git
cd ADE_Classifier-demo
````

### 2. Set Up Python Environment

We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
cd..
cd..
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.1. Train the Classification Model

```bash
## Format: train_classification.py --dataset [choices: ade, psytar] --numsamples [int]
cd trainer
python train_classification.py --dataset ade --numsamples 100
```

### 4.2. Test the Classification Model

```bash
## Format: evaluate_classification.py --model [choices: ade, psytar] --dataset [choices: ade, psytar]
## optional: --model_path relative_path_to_model
cd ../evaluation
python evaluate_classification.py --model ade --dataset psytar
# Or just evaluate on selected model:
python evaluate_classification.py --model ade --dataset psytar --model_path "bert_model_fold_1_set_2"
```

### 5.1 Train the NER Model

```bash
cd trainer
python train_ner.py
```

### 5.2. Test the NER Model

```bash
cd ../evaluation
python evaluate_ner.py
```

## 6. Install & use development dependencies

Prerequisites: Followed [#Quickstart](#quickstart) to set up the environment.

### Additionally install dev dependencies
```bash
pip install -r requirements-dev.txt
```

### Use ruff
[Ruff](https://github.com/astral-sh/ruff) is a python linter and code formatter. 
To run the linter, execute:
```bash
ruff check . --fix
```

To run the code formatter, execute:
```bash
ruff format .
```


---

## 📚 Citation

If you use or reference our work, please cite:

> Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.

> Gurulingappa, H., Rajput, A. M., Roberts, A., Fluck, J., Hofmann-Apitius, M., & Toldo, L. (2012). Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports. *Journal of Biomedical Informatics*, 45(5), 885–892.

> Zolnoori, M., et al. (2019). The PsyTAR dataset: From patients generated narratives to a corpus of adverse drug events and effectiveness of psychiatric medications. *Data in Brief*, 24, 103838.
> Karimi, Sarvnaz; Metke Jimenez, Alejandro; Kemp, Madonna; & Wang, Chen (2015): CADEC. v3. CSIRO. Data Collection. https://doi.org/10.4225/08/570FB102BDAD2

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



