import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

def load_tokenizer_and_model():
    # Load pretrained model from checkpoint
    checkpoint_path = './best_model_ckpt'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()

    # Sanity check
    # Example input text
    try:
        input_text = "Hello, how are you?"
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        print(outputs)
        print(logits)
        print("** Sanity check passed **")
    except:
        print("** SANITY CHECK FAILED **")
    return tokenizer, model

def load_datasets(sample_size=None):
    # From Huggingface
    if sample_size:
        sst2_dataset = load_dataset('glue', 'sst2', split='train[:{}]'.format(sample_size))
        imdb_dataset = load_dataset('imdb', split='train[:{}]'.format(sample_size))
    else:
        sst2_dataset = load_dataset('glue', 'sst2', split='train')
        imdb_dataset = load_dataset('imdb', split='train')
    return sst2_dataset, imdb_dataset

def create_member_non_member(member_dataset, non_member_dataset):
    # Sample member and non-member data
    member_data = member_dataset['sentence']
    member_labels = [1] * len(member_data)  # 1 for members

    # Sample non-member data (using a balanced approach)
    non_member_data = non_member_dataset['text'][:len(member_data)]  # Take an equal amount from alternate distribution
    non_member_labels = [0] * len(non_member_data)  # 0 for non-members

    # Combine member and non-member data
    texts = member_data + non_member_data
    labels = member_labels + non_member_labels
    return texts, labels

class MIADataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def extract_features(model, tokenizer, data_loader, show_progress=False):
    model.eval()
    features, labels = [], []
    confidence_scores, entropy_scores = [], []

    with torch.no_grad():
        data_iter = tqdm(data_loader, desc="Extracting LLM Features") if show_progress else data_loader
        for texts, lbls in data_iter:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            features.append(logits.numpy())
            labels.append(lbls.numpy())
            # Additional features
            # hidden_states = outputs.hidden_states[-1]
            # Softmax to get probabilities
            # probabilities = torch.softmax(logits, dim=1).numpy()
            # confidence_scores.append(np.max(probabilities, axis=1))
            # entropy_scores.append(-np.sum(probabilities * np.log(probabilities + 1e-10), axis=1))

    return np.concatenate(features), np.concatenate(labels)

def train_mia_classifier_rf(X_train, y_train, X_test, y_test, show_progress=False):
    classifier = RandomForestClassifier()
    if show_progress:
        # Use tqdm to show progress bar during training
        classifier.fit(X_train, y_train,
                       callbacks=[tqdm(total=len(X_train), desc="Training MIA Classifier")])
    else:
        classifier.fit(X_train, y_train)

    # Evaluate MIA classifier
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report



def main(show_progress=False):
    print("** Process Member and Non-member data **")
    sst2_dataset, imdb_dataset = load_datasets(sample_size=100)
    texts, labels = create_member_non_member(sst2_dataset, imdb_dataset)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=42)

    # Initialize the tokenizer and model
    tokenizer, model = load_tokenizer_and_model()

    # Create DataLoader for training and validation
    train_dataset = MIADataset(X_train, y_train)
    val_dataset = MIADataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("** Training MIA classifier **")
    # Extract features
    X_train_features, y_train_labels = extract_features(model, tokenizer, train_loader, show_progress)
    X_val_features, y_val_labels = extract_features(model, tokenizer, val_loader, show_progress)

    print("** Evaluating MIA classifier : Random Forest **")
    accuracy, report = train_mia_classifier_rf(X_train_features, y_train_labels, X_val_features, y_val_labels, show_progress)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)


if __name__ == "__main__":
    show_progress = True
    main(show_progress)

