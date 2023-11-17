from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.metrics import accuracy_score
import numpy as np

# Define Classifier


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.batchnorm(x)
        x = self.fc3(x)
        return x


def get_data(data_name):
    if data_name == '20ng':
        train_data = fetch_20newsgroups(
            subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(
            subset='test', remove=('headers', 'footers', 'quotes'))
        train_text, train_labels = train_data['data'], train_data['target']

        # Original test data
        original_test_text, original_test_labels = test_data['data'], test_data['target']

        # Split the original test data into 50% test and 50% validation
        test_text, val_text, test_labels, val_labels = train_test_split(
            original_test_text, original_test_labels, test_size=0.5, random_state=42)

        return train_text, val_text, test_text, train_labels, val_labels, test_labels


# Fetch data
train_text, val_text, test_text, train_labels, val_labels, test_labels = get_data(
    '20ng')


# Assume these are your precomputed embeddings aligned with the fetched text and labels
train_embeddings_topic_model = np.load(
    'save_embeddings_testsplit/doc_topic/all_eval_doc_embeddings_train.npy')
val_embeddings_topic_model = np.load(
    'save_embeddings_testsplit/doc_topic/all_eval_doc_embeddings_val.npy')
test_embeddings_topic_model = np.load(
    'save_embeddings_testsplit/doc_topic/all_eval_doc_embeddings_test.npy')

train_embeddings_avg_doc = np.load(
    'save_embeddings_testsplit/avg_doc_emb/all_doc_embeddings_avg_train.npy')
val_embeddings_avg_doc = np.load(
    'save_embeddings_testsplit/avg_doc_emb/all_doc_embeddings_avg_val.npy')
test_embeddings_avg_doc = np.load(
    'save_embeddings_testsplit/avg_doc_emb/all_doc_embeddings_avg_test.npy')

train_embeddings_bert = np.load(
    'save_embeddings_testsplit/sbert/train_embeddings_bert.npy')
val_embeddings_bert = np.load(
    'save_embeddings_testsplit/sbert/val_embeddings_bert.npy')
test_embeddings_bert = np.load(
    'save_embeddings_testsplit/sbert/test_embeddings_bert.npy')


print('train_text', len(train_text))
print('train_embeddings_topic_model', len(train_embeddings_topic_model))
print('val_text', len(val_text))
print('val_embeddings_topic_model', len(val_embeddings_topic_model))
print('test_text', len(test_text))
print('test_embeddings_topic_model', len(test_embeddings_topic_model))

# Train and Evaluate


def train_and_eval(train_embeddings, val_embeddings, test_embeddings, train_labels, val_labels, test_labels, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Tensor datasets
    train_data = TensorDataset(torch.FloatTensor(
        train_embeddings), torch.LongTensor(train_labels))
    val_data = TensorDataset(torch.FloatTensor(
        val_embeddings), torch.LongTensor(val_labels))
    test_data = TensorDataset(torch.FloatTensor(
        test_embeddings), torch.LongTensor(test_labels))

    print('train shape', train_embeddings.shape)
    print(len(train_labels))
    print('test shape', test_embeddings.shape)
    print(len(test_labels))
    print('val shape', val_embeddings.shape)
    print(len(val_labels))

    # Dataloaders3
    train_loader = DataLoader(train_data, shuffle=True, batch_size=128)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=128)

    model = Classifier(train_embeddings.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    best_val_accuracy = 0
    best_test_accuracy = 0

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        val_loss = 0
        val_accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                val_loss += criterion(output, labels)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_accuracy += torch.mean(equals.type(torch.FloatTensor))

        val_accuracy = val_accuracy/len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Validation accuracy: {val_accuracy.item()*100}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

         # Test Accuracy
        test_accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))
        test_accuracy = test_accuracy / len(test_loader)
        print(f"Test Accuracy for {model_name}: {test_accuracy.item()*100}%")
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy

    print(
        f"Best Validation Accuracy for {model_name}: {best_val_accuracy.item()*100}%")
    print(
        f"Best Test Accuracy for {model_name}: {best_test_accuracy.item()*100}%")


train_and_eval(train_embeddings_topic_model, val_embeddings_topic_model,
               test_embeddings_topic_model, train_labels, val_labels, test_labels, 'Topic_Model')
train_and_eval(train_embeddings_avg_doc, val_embeddings_avg_doc,
               test_embeddings_avg_doc, train_labels, val_labels, test_labels, 'Avg_Doc')
train_and_eval(train_embeddings_bert, val_embeddings_bert,
               test_embeddings_bert, train_labels, val_labels, test_labels, 'BERT')
