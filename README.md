# Lesion Model Observer

A PyTorch-based binary classifier for evaluating lesion detectability using 16-bit PNG phantom images reconstructed with different methods (e.g., FPB, IMAR).

## Features
- ResNet18-based model adapted for single-channel input
- Augmentation: flip, rotation, affine, perspective, blur, intensity shift
- 5-fold cross-validation and ensemble evaluation
- AUC metrics grouped by lesion and dose

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the training and evaluation:
    ```bash
    python train_dlmo.py
3. Optional: On HPC,
    ```bash
    sbatch sbatch_train_dlmo.sh

3. Outputs
    Trained models (*.pt)
    Validation/test metrics (*.txt, *.csv)
    Training curves (*.png)

4. Run the testing:
    ```bash
    Example: python test_dlmo.py --data_dir ./data/MO_CHO_Lesion_png_16bit --lesions Lesion13 Lesion14 Lesion15 Lesion16 --model_path "./results/dlmo_20250424_091921/experiment_1/IMAR/fold_1/model.pt" --method IMAR --out_dir ./

## Data Format
16-bit PNG images named as: SP_Lesion01_Loc01_F.png (SP/SA: signal, F/H: dose)

## Requirements:
    torch>=1.10
    torchvision
    opencv-python
    numpy
    matplotlib
    scikit-learn
    pandas
