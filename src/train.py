import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from data.loaders import create_loaders
from models.mlp import MLPRegressor
from trainers.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rmse_loss(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("dataset/insurance.csv")

    target = "charges"
    categorical_cols = ["sex", "smoker", "region"]
    numerical_cols = [c for c in df.columns if c not in categorical_cols + [target]]

    X_cat = df[categorical_cols]
    X_num = df[numerical_cols]
    y = df[target].values

    cat_encoder = OrdinalEncoder()
    X_cat = cat_encoder.fit_transform(X_cat)

    num_scalar = StandardScaler()
    X_num = num_scalar.fit_transform(X_num)

    X = np.concatenate([X_cat, X_num], axis=1)

    y = np.log1p(df[target].values).astype(np.float32)
    
    train_loader, val_loader = create_loaders(X, y)

    model = MLPRegressor(X.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=rmse_loss,
        device=device,
        epochs=100,
        save_path="checkpoints/mlp_best.pt",
    )

    trainer.fit(train_loader, val_loader)

train()