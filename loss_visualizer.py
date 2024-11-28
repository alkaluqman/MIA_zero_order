import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from mpl_toolkits.mplot3d import Axes3D
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import Adam
import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_metrics import Metric
from loss_landscapes.model_interface.model_wrapper import ModelWrapper


dataset = load_dataset("glue", "sst2")


checkpoint_path = './best_model_ckpt_ft'
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(checkpoint_path)

def initialize_weights(model):
    for name, param in model.named_parameters():
        # if param.requires_grad and param.ndimension() > 1:  # Skip biases
            # Initialize weights with a small random value
        torch.nn.init.normal_(param, mean=0.0, std=1e-3)

initialize_weights(model)  # Apply custom initialization to avoid division by zero error in random_plane

for name, param in model.named_parameters():
    norm = param.norm().item()
    print(f"Parameter: {name}, Norm: {norm}")
    if norm == 0:
        print(f"Warning: {name} has a zero norm!")

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding=True, truncation=True, return_tensors="pt")

train_dataset = dataset['train'].map(tokenize_function, batched=True)
test_dataset = dataset['test'].map(tokenize_function, batched=True)

train_dataset = train_dataset.select(range(10))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
for batch in train_loader:
    print(batch['label'])

class LLMModelWrapper(ModelWrapper):
    def __init__(self, model: torch.nn.Module):
        # Initialize with the RoBERTa model
        super().__init__([model])

    def forward(self, inputs):
        # Forward pass through the RoBERTa model
        encoding = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        model = self.modules[0]  # Get the first (and only) module, which is the RoBERTa model
        return model(input_ids=input_ids, attention_mask=attention_mask)

class RobertaLoss(Metric):
    def __init__(self, loss_fn, inputs, target):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: LLMModelWrapper) -> float:
        # Get the input data
        inputs = self.inputs
        targets = self.target
        print("BBBBB", targets)
        # Forward pass using the model wrapped in SimpleModelWrapper
        outputs = model_wrapper.forward(inputs)
        logits = outputs.logits
        return self.loss_fn(logits, targets).item()

# Wrap the model with SimpleModelWrapper
wrapped_model = LLMModelWrapper(model)

loss_fn = torch.nn.CrossEntropyLoss()
inputs_batch = next(iter(train_loader))

inputs = inputs_batch['sentence']
labels = inputs_batch['label']

metric = RobertaLoss(loss_fn, inputs, labels)
STEPS = 10
loss_data_fin = loss_landscapes.random_plane(wrapped_model, metric, 10, STEPS, normalization='filter', deepcopy_model=True)
print("Visualizing")
print(loss_data_fin)
fig = plt.figure()
ax = plt.axes(projection='3d')
X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('Loss Landscape - FineTuned model')
fig.savefig("loss_landscape_ft.png")