# Deep Learning Model Observer

A PyTorch-based binary classifier with model d' to compare with CHO model d', and AUC of leasion, for evaluating lesion detectability using 16-bit PNG phantom images reconstructed with different methods (e.g., FPB, IMAR).

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the training and evaluation:
    ```bash
    python KOppStyle.py

3. Outputs
    Trained models (*.pt)
    Validation/test metrics (*.txt, *.csv)
    Training curves (*.png)

## Data Format
16-bit PNG images named as: SA_Lesion02_Loc03_F_FBP_001_slice01_of_24.png (SP/SA: signal, F/H: dose)

## Requirements:
    torch>=1.10
    torchvision
    opencv-python
    numpy
    matplotlib
    scikit-learn
    pandas



## Developed by Pegah Ahadian, pahadian@kent.edu
