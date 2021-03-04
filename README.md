# CapsuleNet for Micro-expression Recognition

## Description
> This is the source code for the paper **CapsuleNet for Micro-expression Recognition** joining the second Facial
Micro-Expression Grand Challenge for Micro-expression Recognition Task.
If you find this code useful, please kindly cite the our paper as follows:

```
# Bibtex
@INPROCEEDINGS{Quang2019Capsulenet,
  author={N. V. {Quang} and J. {Chun} and T. {Tokuyama}},
  booktitle={2019 14th IEEE International Conference on Automatic Face   Gesture Recognition (FG 2019)}, 
  title={CapsuleNet for Micro-Expression Recognition}, 
  year={2019},
  volume={},
  number={},
  pages={1-7},}




# Plain text
N. V. Quang, J. Chun and T. Tokuyama, "CapsuleNet for Micro-Expression Recognition," 2019 14th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2019), Lille, France, 2019, pp. 1-7, doi: 10.1109/FG.2019.8756544.
```




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
