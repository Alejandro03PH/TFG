# dataset.py
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def carregador_dades(csv_path: str,
                batch_size_train: int = 32,
                batch_size_test: int  = 64,
                train_ratio: float    = 0.8,
                shuffle_train: bool   = True):
    # ----------------------------------------------------------
    # Primer carreguem les dades des del fitxer CSV
    df = pd.read_csv(csv_path)

    # ----------------------------------------------------------
    # Posem les dades en el format correcte y les convertim a tensors de PyTorch
    # ----------------------------------------------------------
    X = torch.tensor(df[['x', 'y']].values, dtype=torch.float32)   # (N,2)
    Y = torch.tensor(df[['sortida']].values, dtype=torch.float32) # (N,1)

    # ----------------------------------------------------------
    # Separem les dades en conjunts d'entrenament i de prova (80% - 20%)(El percentatje el defineix el train_ratio)
    # ----------------------------------------------------------
    n = X.shape[0]
    perm = torch.randperm(n)

    n_train = int(train_ratio * n)
    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test , Y_test  = X[test_idx ], Y[test_idx ]

    # ----------------------------------------------------------
    # Creem els Datasets i DataLoaders de PyTorch
    # ----------------------------------------------------------
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset  = TensorDataset(X_test , Y_test)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size_train,
                              shuffle=shuffle_train)

    test_loader  = DataLoader(test_dataset,
                              batch_size=batch_size_test,
                              shuffle=False)

    return train_loader, test_loader