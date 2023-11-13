# BDH 6250 DeepNote-GNN, Reproducability
# Authors: vkrishnamurthy, myamasaki

Nov 10 2023

* Project deliverable for Big Data Analytics for Healthcare in Georgia Tech

This is attempt to reproduce code in parts for the paper "DeepNote-GNN: predicting hospital readmission using clinical notes and patient network* by Sara Nouri Golmaei and Xiao Luo.

## Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```
## Preparing the data

The paper used the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset.
Only the `ADMISSIONS.csv` and `NOTEEVENTS.csv` are needed. Download them and

The data are then preprocessed to extract the readmission information. We adopted the script from ClinicalBERT's [repo](https://github.com/kexinhuang12345/clinicalBERT), as was done by the authors of the paper. Run the preprocessing script with the following command:

```bash
python preprocess.py
```

Then, use the pretrained ClinicalBERT to extract the representation for each clinical note. The pretrained weights are available at HuggingFace's model hub ([link](https://huggingface.co/AndyJ/clinicalBERT)). However, the config file is missing the `model_type` key, which will cause an error if directly loaded from the model hub. So you need to download the model and add `"model_type": "bert"` to `config.json`. After that, run the pretrained model using:

```bash
python run_pretrained.py
```

## Training the model

Using `train.py` to train the DeepNote model as well as the baseline models. For example, you can run the following script to train the DeepNote model:

```bash
python train.py configs/deepnote.yml --savename deepnote
```

To run the bag-of-word model, use the following script:

```bash
python run_bow.py
```