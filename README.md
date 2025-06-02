

# DeepFlowState: Demo Repository for *Understanding Deep Learning*

Welcome to the demo repository for the Osnabrück University course *Understanding Deep Learning*, part of the Cognitive Science program.

This repository showcases a practical application of deep learning using **BERT** (Bidirectional Encoder Representations from Transformers), originally introduced by Devlin et al. (2019). It is maintained by the **DeepFlowState** group and is designed to demonstrate our approach to model fine-tuning and transfer learning in the context of pharmacovigilance.

---

## 🧠 Project Overview

The primary goal of this project is to **detect Adverse Drug Reactions (ADR)** in text using a fine-tuned BERT model. ADR detection is a key part of pharmacovigilance—the ongoing monitoring of drug safety in real-world usage.

We fine-tune the pre-trained BERT model on one of the following labeled datasets:

* **ADE Corpus**
* **IMI WEB-RADR**
* **PsyTAR**

Each of these datasets includes labeled examples of drug-related adverse reactions.

---

## 🔬 Research Focus: Transfer Learning & Generalization

Our project explores the generalization capability of BERT through **cross-dataset transfer learning**. Here's what we investigate:

* Can a BERT model fine-tuned on one dataset effectively detect ADRs in entirely different datasets?
* How well does it transfer across **platforms, annotation guidelines, and drug vocabularies**?

We compare performance across:

* A **test split** from the same dataset used for training (in-domain),
* And **two external test datasets** (out-of-domain).

This allows us to evaluate how well the model captures the *underlying concept* of ADR, independent of dataset-specific biases.

---

## 📁 Repository Contents

This repository includes:
* `BERTmodel.py`: The pretrained model BERT gets finetuned on training data
* `index.html`: An entry-point webpage to visualize results or explain the project
* `style.css`: A CSS stylesheet for custom styling
* GitHub Actions workflows: Automate model training, testing, or deployment

---

## 🔧 Getting Started

To get started with the code, clone the repository and follow the setup instructions in `INSTALL.md` (or include them here if applicable).

---

## 📚 Citation

If you use or reference our work, please cite:

> Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
> Dietrich, J., Gattepaille, L. M., Grum, B. A., Jiri, L., Lerch, M., Sartori, D., & Wisniewski, A. (2020). Adverse Events in Twitter–Development of a Benchmark Reference Dataset: Results from IMI WEB-RADR. Drug Safety, 43, 467–478.
> Gurulingappa, H., Rajput, A. M., Roberts, A., Fluck, J., Hofmann-Apitius, M., & Toldo, L. (2012). Development of a benchmark corpus to support the automatic extraction of drug-related adverse effects from medical case reports. Journal of Biomedical Informatics, 45(5), 885–892.
> Zolnoori, M., Fung, K. W., Patrick, T. B., Fontelo, P., Kharrazi, H., Faiola, A., Shah, N. D., Wu, Y. S. S., Eldredge, C. E., Luo, J., Conway, M., Zhu, J., Park, S. K., Xu, K., & Moayyed, H. (2019). The PsyTAR dataset: From patients generated narratives to a corpus of adverse drug events and effectiveness of psychiatric medications. Data in Brief, 24, 103838. 


