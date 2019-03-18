# CapsuleNet for Micro-expression Recognition

## Description
> This is the source code for the project joining the second Facial
Micro-Expression Grand Challenge for Micro-expression Recognition Task.
The challenge is organized in MEGC Workshop in conjunction with FG 2019.
For more information and details, please visit the workshop [website](https://facial-micro-expressiongc.github.io/MEGC2019).


## The log file result
> `result_log.csv`.

## Some missing and invalid clips 

There are 7 clip file which are missing or invalid. 

* The below clips don't exist in the downloaded datasets
```
smic/HS_long/SMIC_HS_E/s03/s3_ne_03 not exists
smic/HS_long/SMIC_HS_E/s03/s3_ne_20 not exists
smic/HS_long/SMIC_HS_E/s04/s4_ne_05 not exists
smic/HS_long/SMIC_HS_E/s04/s4_ne_06 not exists
smic/HS_long/SMIC_HS_E/s09/s9_sur_02 not exists
```

* The invalid clips in which the apex frame out onset-offset duration

```
samm/28/028_4_1
samm/32/032_3_1
```


## Requirements
* Python 3
* PyTorch
* TorchVision
* TQDM

## Usage
Run the following script to reproduce the result in the log file.

```bash
python train_me_loso.py
python get_result_log.py
```

## Project Structure

* `smic_processing.py`: Preprocess the SMIC dataset: detect the apex frames.
* `train_me_loso.py`: Perform LOSO cross-validation on our proposed model.
* `train_me_loso_baseline.py`: Perform LOSO cross-validation on baseline models (ResNet18, VGG11).
* `get_result_log.py`: Write the result log file from the pickle file.
* `capsule`: The package for building CapsuleNet and Loss.

## To-do list
- [ ] Clean the code.
- [ ] Upload the pre-trained models
- [ ] Write better documentation.
