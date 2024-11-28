import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import sys
import urllib
import pickle
import argparse
sys.path.append('../')
from attack_performance import black_box_benchmarks
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import RobertaModel
from datasets import load_dataset

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
        inputs = tokenizer(input_text, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        print(outputs)
        print(logits)
        print("** Sanity check passed **")
    except:
        print("** SANITY CHECK FAILED **")
    return tokenizer, model

class SST2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(SST2Classifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)  # Dropout layer for regularization
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)


    def forward(self, input_ids, attention_mask=None):
        # Forward pass through the RoBERTa model
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Get the CLS token representation
        pooled_output = self.dropout(pooled_output)  # Apply dropout
        return self.classifier(pooled_output)  # Pass through the classifier

def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.LongTensor(i) for i in features])  # Transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:, 0]
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return dataset

def prepare_sst2_data(tokenizer, batch_size=100):
    DATASET_NUMPY = 'data.npz'

    dataset = load_dataset('glue', "sst2")

    train_encodings = tokenizer(dataset['train']['sentence'], truncation=True, padding=True, return_tensors='np')
    test_encodings = tokenizer(dataset['test']['sentence'], truncation=True, padding=True, return_tensors='np')

    train_data = train_encodings['input_ids']  # Token IDs for training
    train_label = np.array(dataset['train']['label'])  # Labels for training

    test_data = test_encodings['input_ids']  # Token IDs for testing
    test_label = np.array(dataset['test']['label'])  # Labels for testing

    np.random.seed(100)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len // 2]
    target_indices = r[train_len // 2:]

    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    # Testing data is half of the original test set
    test_len = test_data.shape[0]
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_test_indices = r[:test_len // 2]
    target_test_indices = r[test_len // 2:]

    shadow_test_data, shadow_test_label = test_data[shadow_test_indices], test_label[shadow_test_indices]
    target_test_data, target_test_label = test_data[target_test_indices], test_label[target_test_indices]

    shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    shadow_train_loader = torch.utils.data.DataLoader(shadow_train, batch_size=batch_size, shuffle=True, num_workers=1)

    shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)
    shadow_test_loader = torch.utils.data.DataLoader(shadow_test, batch_size=batch_size, shuffle=True, num_workers=1)

    target_train = tensor_data_create(target_train_data, target_train_label)
    target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=batch_size, shuffle=True, num_workers=1)

    target_test = tensor_data_create(target_test_data, target_test_label)
    target_test_loader = torch.utils.data.DataLoader(target_test, batch_size=batch_size, shuffle=True, num_workers=1)

    print('Data loading finished')
    return shadow_train_loader, shadow_test_loader, target_train_loader, target_test_loader


def softmax_by_row(logits, T=1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx) / T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp / denominator


def prepare_model_performance(shadow_model, shadow_train_loader, shadow_test_loader,
                              target_model, target_train_loader, target_test_loader):
    def _model_predictions(model, dataloader):
        return_outputs, return_labels = [], []

        for (inputs, labels) in dataloader:
            return_labels.append(labels.numpy())
            outputs = model.forward(inputs) #.cuda())
            return_outputs.append(softmax_by_row(outputs.data.cpu().numpy()))
        return_outputs = np.concatenate(return_outputs)
        return_labels = np.concatenate(return_labels)
        return (return_outputs, return_labels)

    shadow_train_performance = _model_predictions(shadow_model, shadow_train_loader)
    shadow_test_performance = _model_predictions(shadow_model, shadow_test_loader)

    target_train_performance = _model_predictions(target_model, target_train_loader)
    target_test_performance = _model_predictions(target_model, target_test_loader)
    return shadow_train_performance, shadow_test_performance, target_train_performance, target_test_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run membership inference attacks')
    parser.add_argument('--dataset', type=str, default='sst2', help='sst2 or mnli')
    # parser.add_argument('--model-dir', type=str, default='./pretrained_models/purchase_natural',
    #                     help='directory of target model')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size of data loader')
    args = parser.parse_args()

    tokenizer, model = load_tokenizer_and_model()

    if args.dataset == 'sst2':
        model = SST2Classifier(num_classes=2)
        model = torch.nn.DataParallel(model)#.cuda()
        shadow_train_loader, shadow_test_loader, \
            target_train_loader, target_test_loader = prepare_sst2_data(tokenizer, batch_size=args.batch_size)
    else:
        pass


    shadow_train_performance, shadow_test_performance, target_train_performance, target_test_performance = \
        prepare_model_performance(model, shadow_train_loader, shadow_test_loader,
                                  model, target_train_loader, target_test_loader)

    print('Perform membership inference attacks!!!')
    MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance,
                               target_train_performance, target_test_performance, num_classes=100)
    MIA._mem_inf_benchmarks()