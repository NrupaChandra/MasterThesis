# AI-Powered Acceleration of Cut-Cell Integration

This repository contains the codebase, models, and result evaluation scripts for the master's thesis:

**"AI-Powered Acceleration of Cut-Cell Integration"**  
by **Nrupa Chandra Girish Chandra**  
TU Darmstadt · Institute of Fluid Dynamics (FDY)


### Prerequisites

- Python 3.10+
- PyTorch ≥ 2.0
- PyTorch Geometric
- NumPy
- pandas
- matplotlib
- Optuna
- tqdm
- scikit-learn

- A conda environment was used to set up the modules.


*Training and testing*

-FNN
python FNN/FNN_V6/train_fnn.py and FNN/FNN_V6/test_fnn.py

-CNN
python CNN/train_cnn.py and CNN/test_cnn.py

-CNN-FNN Hybrid
python CNN_FNN_Hybrid/train_hybrid.py and CNN_FNN_Hybrid/test_hybrid.py

-GCNN
python GCNN/train_gcnn.py and GCNN/test_gcnn.py

*optuna optimization*
python FNN/FNN_V6/hyperparameter_optuna.py


**Additional utilities**




