# BubbleSync-GAN (Paper Coming Soon)

This repository provides the official implementation of our BubbleSync-GAN paper titled:<br/>  _**"BubbleSync-GAN: Preserving Physical Characteristics Consistency in Unsupervised Image-to-Image Translation
Through Intelligent Physical Features Extraction"**_

![BubbleSync_Github](https://github.com/user-attachments/assets/4b74fc83-1068-465c-9bec-82bba9902579)

> **Note:** this repository implements BubbleSync-GAN's physical-feature-consistency (blob-based) contribution only. A follow-up work, **SequenceSync-GAN**, extends this approach with a temporal-sequence-consistency contribution (a sequence-aware data loader, a temporal discriminator, and a temporal-consistency loss). See that paper/repository for the combined approach.

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
  ├── get_blobs_properties.py      # Physical feature extraction (reference implementation)
  ├── get_blobs_properties_differentiable.py  # Differentiable blob-loss gradient path (see below)
  ├── verify_blob_gradient.py      # Standalone script demonstrating the gradient-flow bug/fix
```

## Usage

### Clone Repository

```bash
git clone https://github.com/Hindawi91/BubbleSync-GAN.git
cd BubbleSync-GAN
```

### Training

#### 1. Download dataset:
<ol type="1">
  <li>Download our <a href="https://www.dropbox.com/scl/fi/0iqury0rhq7v81bu2rmpe/data.rar?rlkey=2a35eenysxl0uq20ou0wea5b5&dl=0" > data </a> to replace the current data folder</li>
  <li>Download our <a href="https://www.dropbox.com/scl/fi/k3oi23tmbu9nrfpezcwxm/base_classifier.rar?rlkey=iobe3kdis949j6xi2e0csn1do&dl=0" > Base Classifier </a> and place it inside the "base_classifier_training/" folder</li>
  <li>Download our best-performing saved checkpoint models below and place them inside the "Boiling/models/" folder:
    <ul>
      <li><a href="#">exp4_cL_mH_sH_seed202, iteration 120000 (best AUC)</a> -- lambda_count=0.01, lambda_mean=1e-8, lambda_std=1e-7, seed=202</li>
      <li><a href="#">exp4_cL_mH_sH_seed202, iteration 200000 (best Balanced Accuracy)</a> -- same configuration as above</li>
    </ul>
  </li>
</ol>

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

(Images can also sit in further subfolders under `DomainA`/`DomainB`, e.g. class-specific subdirectories -- the data loader searches recursively.)

#### 3. CNN Base Classifier Training:

<ol type="1">
  <li>Assuming one of the domains you have is labeled</li>
  <li>Go to the base_classifier_training folder</li>
  <li>In the “DS_CNN_Training.py” file, change the “dataset” variable to the source DS directory, then run the Python script.</li>
  <li>Once training is done, the best model would be saved as a “.keras” file (native Keras 3 format).</li>
  <li>In the “test_DS_on_DS.py” file, change the “dataset” variable to the source DS directory, then run the Python script. Then, run the Python script to test the saved model on the source dataset for sanity check.</li>
</ol>

```bash
$ cd base_classifier_training/
$ python DS_CNN_Training.py
```

#### 4. BubbleSync-GAN Training

Start Training:

```bash
$ bash train.sh
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

## Fixes since the original implementation

A few correctness issues were found and fixed since this code was first written. If you're comparing results against an earlier checkout of this repo, or auditing the implementation, these are worth knowing about:

- **Blob losses now actually train the Generator.** The original `get_blobs_properties()` computes blob statistics via skimage/NumPy operations, then wraps the result in a fresh `torch.tensor(...)`. That tensor has no `grad_fn` -- it's disconnected from the Generator's computation graph -- so the blob count/mean-area/std-area losses contributed **exactly zero gradient** regardless of `lambda_count`/`lambda_mean`/`lambda_std`. `get_blobs_properties_differentiable.py` fixes this with a straight-through estimator: the forward pass still calls the original `get_blobs_properties()` directly (so reported values are byte-identical, not approximated), while the backward pass routes gradients through a differentiable soft-thresholded proxy mask. `verify_blob_gradient.py` is a small standalone script demonstrating the bug empirically against the original function.
- **Missing grayscale conversion.** The Generator/Discriminator are built for 1-channel input, but the original `data_loader.py` never converted images to grayscale, silently feeding 3-channel input where 1 was expected.
- **`create_labels()` call/signature mismatch.** `solver.py`'s `train()` called `create_labels()` with an extra `selected_attrs` argument that both didn't exist as an attribute and wasn't accepted by the function's own signature -- causing a crash before training could start.
- **`test()` only translated the intended direction.** The original `test()` processed every image in the combined test set and always set the target label to domainA-style, meaning already-domainA images were "translated" to a near-identical copy of themselves -- producing uninformative output alongside the real domainB-to-domainA translations. Fixed to skip domainA samples during testing.
- **Deprecated TensorFlow/Keras APIs.** `classification_test.py` and `DS_CNN_Training.py` imported from `keras.layers.core`/`keras.layers.convolutional`, which don't exist in modern Keras 3, and used `model.fit_generator()`, removed in TensorFlow >= 2.11. Updated to current equivalents.
- **`ModelCheckpoint` `.hdf5` saving.** Keras 3 requires checkpoint filepaths to end in `.keras` or `.weights.h5`; `.hdf5` is no longer accepted for saving (loading existing `.hdf5` files still works).
- **Reproducibility.** Added a `--seed` argument wired through to `torch`, `numpy`, `random`, and `cudnn` deterministic settings.
- **Removed hardcoded `CUDA_VISIBLE_DEVICES="0"`**, which could conflict with cluster job schedulers (e.g. Slurm) that assign GPUs per-job.

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
