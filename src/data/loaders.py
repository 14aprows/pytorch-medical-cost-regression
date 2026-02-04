from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import MedicalCostDataset

def create_loaders(X, y, batch_size=32, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_dataset = MedicalCostDataset(X_train, y_train)
    val_dataset = MedicalCostDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader