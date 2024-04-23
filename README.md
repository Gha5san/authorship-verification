---
{}
---
language: en
license: MIT
tags:
- text-classification
- authorship-verification
- PAN
- Siamese-RoBERTa
repo: https://github.com/Gha5san/authorship-verification

---

# Model Card for roberta-av

<!-- Provide a quick summary of what the model is/does. -->

This model is an advanced text classification model designed to determine if two text samples were authored by the same individual.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a RoBERTa model that was fine-tuned on a corpus comprising 30K pairs of text samples. The texts include a diverse array of sources such as emails, news articles, and blog posts, providing a robust foundation for authorship verification.

- **Developed by:** Ghassan Al Kulaibi
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model:** RoBERTa-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/roberta-base
- **Paper or documentation:** https://aclanthology.org/2020.acl-main.197.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

30K pairs of texts drawn from emails, news articles, and blog posts.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - train_batch_size: 16
      - num_epochs: 4
      - warmup_steps: Calculated as 10% of the total training steps

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 4 hours
      - duration per training epoch: 1 hour
      - model size: 450MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

30% of the training data seprated for validation. Another new dev set of 6K was used for testing

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Matthew's Correlation Coefficient: 0.620
      - ROC-AUC Score: 0.810
      - Precision: 0.83
      - Recall: 0.78
      - F1-score: 0.80
      - Accuracy: 81%

### Results

The model achieved an F1-score of 80% and an accuracy of 81% on the validation set.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: Tesla T4

### Software

      - PyTorch: 2.2.2
      - Transformers: 4.40.0
      - Sentence Transformers: 2.7.0
      - Pytorch: Version used for training, aligning with the latest stable release

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any text sample longer than 256 tokens will be truncated, which might lead to loss of critical information essential for making accurate predictions.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The chosen hyperparameters and model architecture were determined through extensive validation to optimize performance for the specific task of authorship verification.

## Environmental Impact

### CO2 Emission Estimate

Based on an estimated power usage of 70 watts for the Tesla T4 and a global average carbon intensity of 450 gCO2eq/kWh, the total training would emit approximately 0.126 kg CO2. 
