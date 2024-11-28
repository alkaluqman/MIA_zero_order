import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset as DDataset
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import random
import attack_utils
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_tokenizer_and_model():
    # Load pretrained model from checkpoint
    checkpoint_path = './best_model_ckpt_ft'
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()

    # Sanity check
    # Example input text
    try:
        input_text = "Hello, how are you?"
        encoding = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        print(outputs)
        print(logits)
        print("** Sanity check passed **")
    except:
        print("** SANITY CHECK FAILED **")
    return tokenizer, model

def load_datasets(dataset_list=[], sample_size=None):
    # From Huggingface
    train_dataset = []
    test_dataset = []

    for name in dataset_list:
        dataset = load_dataset('glue', name) if name == 'sst2' else load_dataset(name)
        train_dataset.append(dataset['train'])
        test_dataset.append(dataset['test'])

    # Unroll the datasets to be able to iterate over each row
    train_dataset = concatenate_datasets(train_dataset)
    test_dataset = concatenate_datasets(test_dataset)

    if sample_size:
        # For TDD
        train_size = int(len(train_dataset) * sample_size)
        test_size = int(len(test_dataset) * sample_size)
        sampled_train_dataset = train_dataset.select(range(train_size))
        sampled_test_dataset = test_dataset.select(range(test_size))
    else:
        sampled_train_dataset = train_dataset
        sampled_test_dataset = test_dataset

    return sampled_train_dataset, sampled_test_dataset

def create_member_non_member(member_dataset, non_member_dataset, random_seed=123):
    # Sample member data
    member_data = member_dataset['sentence']
    member_labels = [1] * len(member_data)  # 1 for members
    non_member_data = non_member_dataset['sentence']
    non_member_labels = [0] * len(non_member_data)  # 0 for non-members

    # Balance the classes
    member_dataset = DDataset.from_dict({'text': member_data, 'label': member_labels})
    non_member_dataset = DDataset.from_dict({'text': non_member_data, 'label': non_member_labels})
    min_size = min(len(member_dataset), len(non_member_dataset))
    balanced_member_dataset = member_dataset.shuffle(seed=random_seed).select(range(min_size))
    balanced_non_member_dataset = non_member_dataset.shuffle(seed=random_seed).select(range(min_size))
    balanced_dataset = concatenate_datasets([balanced_member_dataset, balanced_non_member_dataset])
    shuffled_balanced_dataset = balanced_dataset.shuffle(seed=random_seed)

    # Combine member and non-member data
    texts = shuffled_balanced_dataset['text']
    labels = shuffled_balanced_dataset['label']
    return texts, labels

class MIADataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class ShadowModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ShadowModel, self).__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

def extract_features(model, tokenizer, data_loader, show_progress=False):
    model.eval()
    features, labels = [], []
    confidence_scores, entropy_scores, hidden_states = [], [], []
    model.config.output_hidden_states = True

    with torch.no_grad():
        data_iter = tqdm(data_loader, desc="Extracting LLM Features") if show_progress else data_loader
        for texts, lbls in data_iter:
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            features.append(logits.numpy())
            labels.append(lbls.numpy())
            # Additional features
            # hidden_states.append(outputs.hidden_states[-1].numpy())
            # Softmax to get probabilities
            probabilities = torch.softmax(logits, dim=1).numpy()
            confidence_scores.append(np.max(probabilities, axis=1))
            entropy_scores.append(-np.sum(probabilities * np.log(probabilities + 1e-10), axis=1))

    # Horizontally stack features, confidence scores, and entropy scores
    combined_features = np.hstack((np.concatenate(features),
                                   np.concatenate(hidden_states),
                                   np.concatenate(confidence_scores)[:, np.newaxis],
                                   np.concatenate(entropy_scores)[:, np.newaxis]))

    return combined_features, np.concatenate(labels)

def prepare_data(tokenizer, texts, labels, max_length=512):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels)

def fine_tune_shadow_model(model, train_data, train_attention_mask, train_labels, epochs=3, batch_size=16):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        for i in range(0, len(train_data), batch_size):
            batch_input_ids = torch.tensor(train_data[i:i + batch_size]).to(device)
            batch_attention_mask = torch.tensor(train_attention_mask[i:i + batch_size]).to(device)
            batch_labels = torch.tensor(train_labels[i:i + batch_size]).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

def membership_inference(shadow_model, data, labels, threshold=0.5):
    shadow_model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        input_ids, attention_mask = data
        input_ids.to(device)
        attention_mask.to(device)
        outputs = shadow_model(input_ids, attention_mask)
        logits = outputs.logits

        # Get predicted probabilities for the positive class
        predicted_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        # Convert probabilities to binary predictions using the threshold
        predicted_classes = (predicted_probs >= threshold).astype(int)

        predictions.extend(predicted_classes)
        true_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Print classification report
    print(f'Classification Report: ')
    print(classification_report(true_labels, predictions))

    # Calculate ROC AUC
    roc_auc = roc_auc_score(true_labels, predicted_probs)
    print(f'ROC AUC: {roc_auc:.4f}')

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    print(f'FPR: ', fpr)
    print(f'TPR: ', tpr)
    print(f'THRESHOLDS: ', thresholds)

    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.4f})')
    # plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()

    # binary classification
    membership_results = np.argmax(predictions, axis=1)
    return membership_results == labels



def main(show_progress=False, random_seed=123):
    print("** Process Member and Non-member data **")
    if show_progress:
        mem_dataset, nonmem_dataset = load_datasets(dataset_list=["sst2"], sample_size=0.01)
    else:
        mem_dataset, nonmem_dataset = load_datasets(dataset_list=["sst2"])
    texts, labels = create_member_non_member(mem_dataset, nonmem_dataset, random_seed=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=random_seed)

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

    print("** Evaluating MIA classifier : Non NN **")
    accuracy, report = attack_utils.train_mia_classifier_rf(X_train_features, y_train_labels, X_val_features, y_val_labels, show_progress)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    print("*"*100)
    print("** Evaluating MIA classifier : NN **")
    accuracy, report = attack_utils.train_mia_classifier_ffn(X_train_features, y_train_labels, X_val_features,
                                                            y_val_labels, show_progress)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

def main1(show_progress=False, random_seed=123):
    print("** Process Member and Non-member data **")
    if show_progress:
        mem_dataset, nonmem_dataset = load_datasets(dataset_list=["sst2"], sample_size=0.01)
    else:
        mem_dataset, nonmem_dataset = load_datasets(dataset_list=["sst2"])
    texts, labels = create_member_non_member(mem_dataset, nonmem_dataset, random_seed=random_seed)

    # Initialize the tokenizer and model
    tokenizer, model = load_tokenizer_and_model()
    model.to(device)

    # Prepare data
    input_ids, attention_mask, labels_tensor = prepare_data(tokenizer, texts, labels)

    # Split into training and testing
    train_input_ids, test_input_ids, train_attention_mask, test_attention_mask, train_labels, test_labels = train_test_split(
        input_ids.numpy(), attention_mask.numpy(), labels, test_size=0.2, random_state=42)

    # Create shadow models
    num_shadow_models = 5  # Adjust
    shadow_models = []
    checkpoint_dir = './best_model_ckpt_ft'

    for _ in range(num_shadow_models):
        shadow_model = ShadowModel(checkpoint_dir, num_classes=2)
        shadow_model.to(device)
        fine_tune_shadow_model(shadow_model, train_input_ids, train_attention_mask, train_labels)
        shadow_models.append(shadow_model)

    # Perform membership inference
    for shadow_model in shadow_models:
        results = membership_inference(shadow_model, (test_input_ids, test_attention_mask), test_labels)
        print(f"Membership inference results: {results}")


if __name__ == "__main__":
    RANDOM_SEED = random.seed(42) #[5,42,103,2048]
    show_progress = False

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    # main(show_progress, RANDOM_SEED)
    main1(show_progress, RANDOM_SEED)
