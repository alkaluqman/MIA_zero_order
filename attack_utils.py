from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from attack_performance import black_box_benchmarks
from tqdm import tqdm

def train_mia_classifier_rf(X_train, y_train, X_test, y_test, show_progress=False):
    if show_progress:
        classifier = RandomForestClassifier(verbose=2)
        classifier.fit(X_train, y_train)
    else:
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

    # Evaluate MIA classifier
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Non-member", "Member"])

    # mia_run = black_box_benchmarks(X_test, y_pred, y_train, y_test, num_classes=2)
    # mia_run._mem_inf_benchmarks()
    return accuracy, report

class MIAClassifier(nn.Module):
    def __init__(self, input_size):
        super(MIAClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x
def train_mia_classifier_ffn(X_train, y_train, X_test, y_test, show_progress=False):
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]  # Number of features
    model = MIAClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    for epoch in tqdm(range(1200), desc="Training FFN", disable=not show_progress):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if show_progress:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    # Evaluate MIA classifier
    with torch.no_grad():
        model.eval()
        y_pred_probs = model(X_test_tensor)
        _, y_pred = torch.max(y_pred_probs, 1)

    accuracy = accuracy_score(y_test, y_pred.numpy())
    report = classification_report(y_test, y_pred.numpy(), target_names=["Non-member", "Member"])

    return accuracy, report