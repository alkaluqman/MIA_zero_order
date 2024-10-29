import torch
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset as DDataset
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import random
import attack_utils

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

    print("** Evaluating MIA classifier : Random Forest **")
    accuracy, report = attack_utils.train_mia_classifier_rf(X_train_features, y_train_labels, X_val_features, y_val_labels, show_progress)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

    print("*"*100)
    print("** Evaluating MIA classifier : Feed forward NN **")
    accuracy, report = attack_utils.train_mia_classifier_ffn(X_train_features, y_train_labels, X_val_features,
                                                            y_val_labels, show_progress)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)


if __name__ == "__main__":
    RANDOM_SEED = random.seed(42) #[5,42,103,2048]
    show_progress = False
    main(show_progress, RANDOM_SEED)

