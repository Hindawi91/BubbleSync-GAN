# BubbleSync-GAN (Paper Coming Soon)

This repository provides the official implementation of our BubbleSync-GAN paper titled:<br/>  _**"BubbleSync-GAN: Preserving Physical Characteristics Consistency in Unsupervised Image-to-Image Translation
Through Intelligent Physical Features Extraction"**_

![BubbleSync_Github](https://github.com/user-attachments/assets/4b74fc83-1068-465c-9bec-82bba9902579)


## Paper

[**Coming Soon**]  <!--(https://www.sciencedirect.com/science/article/abs/pii/S0952197623014392)-->

[Firas Al-Hindawi](https://firashindawi.com)<sup>1</sup>, [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>2</sup>, Abhidnya Patharkar<sup>2</sup>, JiaJing Huang<sup>3</sup>, Teresa Wu<sup>2</sup>, [Han Hu](https://scholar.google.com/citations?user=5RgSI9EAAAAJ&hl=en)<sup>4</sup><br/>

<sup>1</sup>King Fahd University of Petroleum & Minerals (KFUPM);<sup>2</sup>Arizona State University; <sup>3</sup>kennesaw state university<br/>; <sup>4</sup>University of Arkansas<br/>

## Abstract

Accurate detection of the critical heat flux (CHF) in boiling heat transfer is vital for ensuring the safety and reliability of thermal systems. Image-based, non-intrusive CHF detection models have emerged as powerful tools for improving the monitoring and design of heat exchangers. However, their generalizability across experimental setups remains limited due to domain shifts in imaging conditions and physical configurations. To address this challenge, this study introduces BubbleSync-GAN, a novel unsupervised image-to-image translation framework designed to enhance cross-domain CHF classification by preserving physical property consistency during domain translation. The proposed model extracts bubble-level physical characteristics from boiling images and employs three newly introduced domain-guided loss functions, Blob Count Loss, Blob Mean Area Loss, and Blob Standard Deviation Area Loss, to incentivize the generator network to maintain physical property consistency between input and translated images. Experiments show that BubbleSync-GAN outperforms existing cross-domain CHF detection methods, achieving up to 16.7\% higher Balanced Accuracy and 22.6\% higher AUC across domains. An ablation study further confirms that jointly enforcing these physical constraints yields the best overall performance. Beyond CHF detection, BubbleSync-GAN offers a generalizable framework for cross-domain image translation tasks involving bubble/blob based features, such as biomedical imaging of kidney glomeruli or cell morphologies. 

---

## Repository Structure

```text
BubbleSync_GAN/
  ├── base_classifier_training/    # CNN base classifier training
  ├── Boiling/                     # Domain specific experiments and models
  ├── data/                        # Dataset directory (not included)
  ├── data_loader.py               # Data loading utilities
  ├── model.py                     # Generator and discriminator definitions
  ├── solver.py                    # Training logic and loss functions
  ├── main.py                      # Main training entry point
  ├── train.sh                     # Training script
  ├── test.sh                      # Evaluation script
  ├── classification_test.py       # Cross domain classification testing
  ├── logger.py                    # Logging utilities
  ├── get_blobs_properties.py      # Physical feature extraction
```

## Usage

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/BubbleSync-GAN.git
cd BubbleSync-GAN
```

### Training

#### 1. Download dataset:
<ol type="1">
  <li>Download our <a href="https://www.dropbox.com/scl/fi/0iqury0rhq7v81bu2rmpe/data.rar?rlkey=2a35eenysxl0uq20ou0wea5b5&dl=0" > data </a> to replace the current data folder</li>
  <li>Download our <a href="https://www.dropbox.com/scl/fi/k3oi23tmbu9nrfpezcwxm/base_classifier.rar?rlkey=iobe3kdis949j6xi2e0csn1do&dl=0" > Base Classifier </a> and place it inside the "base_classifier_training/" folder</li>
  <!-- <li>Download our <a href="https://www.dropbox.com/scl/fi/vyf26trwrx509knfby1pz/models.rar?rlkey=k0qdmrljrek5cpfszvj9osua1&dl=0" > Saved Checkpoint Models </a> and place them inside the "boiling/models/" folder. (best BA model @ 150000, best AUC model @ 90000</li>
</ol> -->

#### 2. Data Preparation

The folder structure should be as follows:

```python
├─data/ # data root
│ ├─train   # directory for training data
│ │ ├─DomainA   # DomainA Train Images
│ │ │ ├─xxx.jpg
│ │ │ ├─ ......
│ │ ├─DomainB   # DomainB Train Images
│ │ │ ├─yyy.jpg
│ │ │ ├─ ......
│ ├─val   # directory for val data
│ │ ├─DomainA   # DomainA val Images
│ │ │ ├─xxx.jpg
│ │ │ ├─ ......
│ │ ├─DomainB   # DomainB val Images
│ │ │ ├─yyy.jpg
│ │ │ ├─ ......
│ ├─test   # directory for test data
│ │ ├─DomainA   # DomainA test Images
│ │ │ ├─xxx.jpg
│ │ │ ├─ ......
│ │ ├─DomainB   # DomainB test Images
│ │ │ ├─yyy.jpg
│ │ │ ├─ ......
```

#### 3. CNN Base Classifier Training:

<ol type="1">
  <li>Assuming one of the domains you have is labeled</li>
  <li>Go to the base_classifier_training folder</li>
  <li>In the “DS_CNN_Training.py” file, change the “dataset” variable to the source DS directory, then run the Python script.</li>
  <li>Once training is done, the best model would be saved as “CNN - Base Model.hdf5”</li>
  <li>In the “test_DS_on_DS.py” file, change the “dataset” variable to the source DS directory, then run the Python script. Then, run the Python script to test the saved model on the source dataset for sanity check.</li>
</ol>

```bash
$ cd base_classifier_training/
$ python DS_CNN_Training.py
```

#### 4. BubbleSync-GAN Training

Start Training:

```bash
$ bash run.sh
```

#### 4. BubbleSync-GAN test Data Translation

Once Training is done, you need to generate results from each checkpoint model saved

```bash
$ bash test.sh
```

#### 5. BubbleSync-GAN Cross-Domain Classification Testing

Once image translation is done, you need to test cross domain classification from each checkpoint model (Assuming you already have a pre-trained classifier on domain A, other wise go to the CNN Base Classifier Training step below): 

```python
$ python classification_test.py
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{BubbleSyncGAN,
  title={BubbleSync-GAN: Preserving Physical Characteristics Consistency in Unsupervised Image-to-Image Translation Through Intelligent Physical Features Extraction},
  author={Al-Hindawi, Firas and Siddiquee, Md Mahfuzur Rahman and others},
  journal={To appear},
  year={2025}
}
```








