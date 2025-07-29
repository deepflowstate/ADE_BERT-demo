
# DeepFlowState: Demo Repository for *Understanding Deep Learning*

Welcome to the demo repository for the Osnabrück University course *Understanding Deep Learning*, part of the Cognitive Science program.

This repository showcases a practical application of deep learning using **BERT** (Bidirectional Encoder Representations from Transformers), originally introduced by Devlin et al. (2019). It is maintained by the **DeepFlowState** group and is designed to demonstrate our approach to model fine-tuning and transfer learning in the context of pharmacovigilance.

---

## 🧠 Project Overview

The primary goal of this project is to **detect Adverse Drug Reactions (ADR)** in text using a fine-tuned BERT model. ADR detection is a key part of pharmacovigilance—the ongoing monitoring of drug safety in real-world usage. It is important to detect adverse reactions to medicine as early as possible. Tools that can be used for this early detection are NLP methods applied on posts on social media and medical feedback forums.

We fine-tune and test the pre-trained BERT model on the following labeled datasets:

- **ADE Corpus**: Data from general Medical Case Reports (general medicine)
- **IMI WEB-RADR**: Data from Twitter posts (general medicine)
- **PsyTAR**: Data from medical feedback forums (specific to psychiatry)
- **CADEC**: Data from medical feedback forums (general medicine)

Each of these datasets includes labeled examples of drug-related adverse reactions.

---

## 🔬 Novelty & Research Focus: Transfer Learning & Generalization

Our project explores the generalization capability of BERT through **cross-dataset transfer learning**. This constitutes the novelty of our approach that we go beyond 'in-domain' training & testing and we train 'in-domain' and test 'out-of-domain' Here's what we investigate:

- Can a BERT model fine-tuned on one dataset effectively detect ADRs in entirely different datasets? (E.g. from a general medicine dataset like ADE Corpus to a specific psychiatry dataset like PsyTAR)
- How well does it transfer across **platforms, annotation guidelines, and drug vocabularies**?

We compare performance across:

- A **training dataset** from one dataset (ADE Corpus) used for training (in-domain)
- And **an external test dataset** (PsyTAR) (out-of-domain)

This allows us to evaluate how well the model captures the *underlying concept* of ADR, independent of dataset-specific biases. We plan to train and test with all datasets crosswise.

---

## 📁 Repository Contents

This repository includes:

- 'python_scripts/':
- `BERTmodel.py`: The pretrained model BERT and specifications (working file)
- `BERTmodelold.py`: Old version of the model. We plan to rework this (old version)
- `kfold_controller.py`: Training the BERT model using 3-fold cross-validation
- `evaluate.py`: File to test the finetuned BERT model (currently on PsyTAR)
`data_sets/`: The datasets we use for training and testing
- 'jupyter_notebooks/':
- `ade_param_comparison.ipynb`: Visualisation and plots of results
  
- `requirements.txt`: required versions of packages and programs that need to be installed to run this repo
- 'requirements-dev.txt': requirded verions of packages and programs for developers
- `dependencies.txt`: File where we want to collect dependencies (is currently not in use)
- `.gitignore`: File to indicate github what should be ignored
- `README.md`: The file you are currently reading

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
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python kfold_controller_ADE_Corpus.py
```

### 5. Test the Model

```bash
python evaluate.py
```

## 4. Install & use development dependencies

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

> Dietrich, J., Gattepaille, L. M., Grum, B. A., Jiri, L., Lerch, M., Sartori, D., & Wisniewski, A. (2020). Adverse Events in Twitter–Development of a Benchmark Reference Dataset: Results from IMI WEB-RADR. *Drug Safety*, 43, 467–478.

> Gurulingappa, H., Rajput, A. M., Roberts, A., Fluck, J., Hofmann-Apitius, M., & Toldo, L. (2012). Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports. *Journal of Biomedical Informatics*, 45(5), 885–892.

> Zolnoori, M., et al. (2019). The PsyTAR dataset: From patients generated narratives to a corpus of adverse drug events and effectiveness of psychiatric medications. *Data in Brief*, 24, 103838.
> Karimi, Sarvnaz; Metke Jimenez, Alejandro; Kemp, Madonna; & Wang, Chen (2015): CADEC. v3. CSIRO. Data Collection. https://doi.org/10.4225/08/570FB102BDAD2

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



