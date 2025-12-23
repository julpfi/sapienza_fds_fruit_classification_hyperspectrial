# Sapienza FDS Project - Hyperspectral Fruit Classification

## Information
1. Course: Fundamentals of Data Science (9 CFU)
2. Team: Julius Pfingsten, Ludovico Piras, and Nicolò Boscherini

## References
This project builds upon the data and methodologies established in the following works:

1. **Varga, L. A., Makowski, J., & Zell, A. (2021).** *Measuring the ripeness of fruit with hyperspectral imaging and deep learning.* IEEE Xplore.
2. **Jmaa, A.B., Chaieb, F., & Fabijańska, A. (2025).** *Fruit-HSNet: A Machine Learning Approach for Hyperspectral Image-Based Fruit Ripeness Prediction.* International Conference on Agents and Artificial Intelligence.

## Description
This project utilizes Hyperspectral Imaging (HSI) to overcome the drawbacks of traditional, invasive fruit ripeness testing methods. By applying Attention-based Deep Learning models and specialized band reduction techniques to high-dimensional spectral data, we achieve accurate, non-destructive classification of fruits into Unripe, Ripe, and Overripe stages.

---

## Dataset
We utilized the **DeepHS Fruit dataset** (Varga et al., 2021). It includes hyperspectral images of five different fruit types, captured by three distinct hyperspectral cameras (covering different wavelengths). Later, we implemented a custom split into test/val/train sets to ensure the front/back pictures are in the same set.

## Methodology

### Band Reduction Strategies
To reduce computational effort and extract relevant spectral features, we implemented four band reduction strategies:

* **Fixed-Average:** The spectrum is split into k equally sized bins and all channels within these bins are averaged.
* **Gaussian-Weighted Average:** Overlapping Gaussian distributions cover the spectrum. Each output channel is a weighted sum of the original bands, determined by the Gaussian curve.
* **Magnitude Low-Pass Filter (DFT):** Uses Discrete Fourier Transform to decompose signatures. Retains the magnitude of the k lowest frequency components.
* **Magnitude and Shift Low-Pass Filter (DFT):** Similar to the magnitude filter but preserves the phase shift, encoding the precise location of spectral features.

### Architectures
We investigated four primary architectures, comparing models trained from scratch against those using transfer learning:

1. **Hybrid-CNN:** A fully trainable model combining spatial and sequential learning. It extracts spatial features using two 3x3 CNN layers, reduces dimensions via pooling, and processes patches with a 1-layer Transformer encoder.
2. **Spectral-NN:** Designed to process spectral signatures directly. It applies global average pooling to the input, followed by a 2-layer 1D CNN feature extractor and a 2-layer Transformer encoder.
3. **Swin-Based:** Adapts a pre-trained Swin-Tiny model. It replaces the input embedding with a custom 4x4 convolutional projection to handle hyperspectral channels. The backbone is frozen; only the adapter, normalization layers, and head are trained.
4. **DeiT-Based:** Adapts the Data-efficient Image Transformer (DeiT-Tiny) using a 16x16 convolutional input adapter. It leverages pre-trained distilled tokens for robust inference. The backbone is frozen; only the adapter, normalization layers, and head are trained.

### Experimental Setup
To ensure a fair comparison, we established a standardized training protocol for all models:
* **Epochs:** 25
* **Optimizer:** AdamW
* **Learning Rate:** 1e-4
* **Loss Function:** Cross-Entropy Loss

---

## Usage and Installation

### Prerequisites
* Python 3.8+
* PyTorch
* Spectral
* WandB (for logging)

### Installation
Clone the repository and install the dependencies:

```bash
git clone <repository_url>
cd sapienza_fds_fruit_classification_hyperspectrial
pip install -r requirements.txt
```

### Configuration
The training pipeline is strictly controlled via the [Config file](src/config.py). You do not need to pass command-line arguments. Instead, open the file and modify the CONFIG dictionary to set up your experiment:

* Fruit Selection: Change fruit to the desired FruitType (e.g., FruitType.KIWI, FruitType.AVOCADO).
* Model Selection: Update model_type to choose the architecture (e.g., 'deit', 'swin', 'fruiths_net', 'hybrid').
* Band Reduction: Set band_reduction to your desired strategy (e.g., 'all', 'uniform', 'dft', 'gaussian_average').
* Hyperparameters: You can also tune batch_size, epochs, and lr (learning rate) directly in this file.

### Downloading the Data 
The data provided by Varga et al. (2021) can be downloaded here: 
[DeepHS Fruit Datasets](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/)

We have stored the data inside our Google drive inside the data folder.

### Running the Training
For execution on Google Colab, ensure the data is mounted on Google Drive. The training can then be set up and started using the notebook [Train_colab_gpu](scripts/train_colab_gpu.ipynb) stored inside the root of the Google drive folder. 

The script will automatically load the configuration, initialize the selected model and dataset, and begin the training loop. Logging is handled via Weights & Biases (requires an API key).