# CapsuleNet for Micro-expression Recognition

## The log file result
> `result_log.csv`.

## Description
> This is the source code for the project joining the second Facial
Micro-Expression Grand Challenge for Micro-expression Recognition Task.
The challenge is organized in MEGC Workshop in conjunction with FG 2019.
For more information and details, please visit the workshop [website](https://facial-micro-expressiongc.github.io/MEGC2019).

## Requirements
* Python 3
* PyTorch
* TorchVision
* TorchNet
* TQDM
* Visdom

## Usage
Run the following script to reproduce the result in the log file.

```bash
python train_me_loso.py
```

## Project Structure

* `smic_processing.py`: Preprocess the SMIC dataset: detect the apex frames.
* `train_me_loso.py`: Perform LOSO cross-validation on our proposed model.
* `train_me_loso_baseline.py`: Perform LOSO cross-validation on baseline models (ResNet18, VGG11).
