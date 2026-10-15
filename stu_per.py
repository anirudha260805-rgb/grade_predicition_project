




from email import errors

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


DATA_PATH = "Student_Performance.csv"
MODEL_PATH = "student_score_regression_model.pth"


class Scorepredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, inputs):
        return self.model(inputs)


def load_training_data(data_path=DATA_PATH):
    data = pd.read_csv(data_path)
    
    
    data["overall_score"] = (data["math_score"] + data["science_score"] + data["english_score"]) / 3
    if "overall_score" not in data.columns:
        raise ValueError(
            "Student_Performance.csv must contain an 'overall_score' column."
        )

    return data.dropna(subset=["overall_score"]).reset_index(drop=True)


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    data = load_training_data(data_path)

    # Exclude the target and the subject scores used to calculate it.
    features = data.drop(
        columns=[
            "overall_score",
            "final_grade",
            "student_id",
            "math_score",
            "science_score",
            "english_score",
        ],
        errors="ignore"
    )

    features = pd.get_dummies(features, dummy_na=True)

    # Predict the numeric score directly instead of treating every score as a class.
    X = features.to_numpy(dtype=np.float32)
    y = data["overall_score"].to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle missing values
    feature_medians = np.nanmedian(X_train, axis=0)
    feature_medians = np.nan_to_num(
        feature_medians, nan=0.0
    )

    X_train = np.where(np.isnan(X_train), feature_medians, X_train)
    X_test = np.where(np.isnan(X_test), feature_medians, X_test)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())
    if target_scale == 0:
        target_scale = 1.0
    y_train_scaled = (y_train - target_mean) / target_scale

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_scaled[:, None], dtype=torch.float32),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False
    )

    torch.manual_seed(42)
    model = Scorepredictor(
        X_train.shape[1], 1
    ).to(device)

    loss_function = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.00001
    )

    epochs = 500

    # Train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_features)
            loss = loss_function(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Loss: {avg_loss:.4f}"
            )

    # Evaluate continuous predictions on unseen test data.
    model.eval()
    predictions = []

    with torch.no_grad():
        for test_features, test_labels in test_loader:
            test_features = test_features.to(device)
            scaled_predictions = model(test_features).squeeze(1)
            predictions.append(
                scaled_predictions.cpu().numpy() * target_scale + target_mean
            )

    predictions = np.concatenate(predictions)
    errors = predictions - y_test
    mae = mean_absolute_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    
    within_tolerance = float(np.mean(np.abs(errors) <= 0.5))
    print(f"Test MAE: {mae:.2f} points")
    print(f"Test R²: {r_squared:.4f}")
    print(f"Test predictions within ±0.5 points: {within_tolerance * 100:.2f}%")
    print("\n----- ERROR ANALYSIS -----")
    print("Total test records:", len(y_test))
    
    print("Within ±1:", np.mean(np.abs(errors) <= 1.0) * 100, "%")
    print("Within ±2:", np.mean(np.abs(errors) <= 2.0) * 100, "%")
    print("Within ±3:", np.mean(np.abs(errors) <= 3.0) * 100, "%")

    # Save the model and preprocessing information
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": X_train.shape[1],
        "feature_columns": features.columns.tolist(),
        "original_feature_columns": features.columns.tolist(),
        "target_mean": torch.tensor(target_mean, dtype=torch.float32),
        "target_scale": torch.tensor(target_scale, dtype=torch.float32),
        "feature_medians": torch.tensor(
            feature_medians, dtype=torch.float32
        ),
        "scaler_mean": torch.tensor(
            scaler.mean_, dtype=torch.float32
        ),
        "scaler_scale": torch.tensor(
            scaler.scale_, dtype=torch.float32
        ),
    }

    torch.save(checkpoint, model_path)
    print(f"Saved trained model to {model_path}")

    return mae


def load_saved_model(model_path=MODEL_PATH):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model = Scorepredictor(
        checkpoint["input_size"],
        1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, device


def preprocess_input_features(raw_features, checkpoint):
    if isinstance(raw_features, dict):
        features = pd.DataFrame([raw_features])
    else:
        features = pd.DataFrame(raw_features)

    features = features.drop(
        columns=["student_id", "overall_score", "final_grade"],
        errors="ignore"
    )

    features = pd.get_dummies(features, dummy_na=True)

    features = features.reindex(
        columns=checkpoint["feature_columns"],
        fill_value=0
    )

    feature_values = features.to_numpy(dtype=np.float32)

    feature_medians = (
        checkpoint["feature_medians"].cpu().numpy()
    )

    feature_values = np.where(
        np.isnan(feature_values),
        feature_medians,
        feature_values
    )

    scaler_mean = checkpoint["scaler_mean"].cpu().numpy()
    scaler_scale = checkpoint["scaler_scale"].cpu().numpy()

    return (
        (feature_values - scaler_mean) / scaler_scale
    ).astype(np.float32)


def predict_score(raw_features, model_path=MODEL_PATH):
    model, checkpoint, device = load_saved_model(model_path)

    scaled_features = preprocess_input_features(
        raw_features, checkpoint
    )

    with torch.no_grad():
        inputs = torch.tensor(
            scaled_features, dtype=torch.float32
        ).to(device)

        predicted_scaled_score = (
            model(inputs).squeeze(1).cpu().numpy()[0]
        )

    score = predicted_scaled_score * checkpoint["target_scale"].item()
    score += checkpoint["target_mean"].item()
    return float(np.clip(score, 0, 100))


if __name__ == "__main__":
    train_model()
