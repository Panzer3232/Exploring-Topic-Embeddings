import torch
import numpy as np
from data.get_data import get_data, get_vocab_embeddings

# Get DataLoader and vocab
batch_size = 128
train_loader, test_loader, val_loader, train_labels, test_labels, val_labels, text, vocab = get_data(
    '20ng', batch_size)

vocab_embeddings = get_vocab_embeddings(vocab)


def get_doc_embeddings_from_bow(batch_bow, vocab_embeddings):
    batch_doc_embeddings = []
    for doc_bow in batch_bow:
        doc_word_embeddings = vocab_embeddings[doc_bow > 0]
        if doc_word_embeddings.shape[0] > 0:
            doc_embedding = torch.mean(doc_word_embeddings, dim=0)
        else:
            doc_embedding = torch.zeros(vocab_embeddings.shape[1])
        batch_doc_embeddings.append(doc_embedding)
    return torch.stack(batch_doc_embeddings)


# Initialize empty lists for storing labels
all_labels_train = []
all_labels_test = []
all_labels_val = []

# For train_loader
all_doc_embeddings_train = []
for batch in train_loader:
    batch_bow = batch['bow'].float()
    all_labels_train.extend(batch['label'].numpy())
    batch_doc_embeddings = get_doc_embeddings_from_bow(
        batch_bow, vocab_embeddings)
    all_doc_embeddings_train.append(batch_doc_embeddings)
all_doc_embeddings_train = torch.cat(all_doc_embeddings_train, dim=0)
print('Train-labels', all_labels_train)
np.save('save_embeddings/all_doc_embeddings_avg_train.npy',
        all_doc_embeddings_train.numpy())
np.save('save_embeddings/save_labels/all_labels_train.npy',
        np.array(all_labels_train))
print('Train shape:', all_doc_embeddings_train.shape)

# For test_loader
all_doc_embeddings_test = []
for batch in test_loader:
    batch_bow = batch['bow'].float()
    all_labels_test.extend(batch['label'].numpy())
    batch_doc_embeddings = get_doc_embeddings_from_bow(
        batch_bow, vocab_embeddings)
    all_doc_embeddings_test.append(batch_doc_embeddings)
all_doc_embeddings_test = torch.cat(all_doc_embeddings_test, dim=0)
print('Test-labels', all_labels_test)
np.save('save_embeddings/all_doc_embeddings_avg_test.npy',
        all_doc_embeddings_test.numpy())
np.save('save_embeddings/save_labels/all_labels_test.npy',
        np.array(all_labels_test))
print('Test shape:', all_doc_embeddings_test.shape)

# For val_loader
all_doc_embeddings_val = []
for batch in val_loader:
    batch_bow = batch['bow'].float()
    all_labels_val.extend(batch['label'].numpy())
    batch_doc_embeddings = get_doc_embeddings_from_bow(
        batch_bow, vocab_embeddings)
    all_doc_embeddings_val.append(batch_doc_embeddings)
all_doc_embeddings_val = torch.cat(all_doc_embeddings_val, dim=0)
print('val-labels', all_labels_val)
np.save('save_embeddings/all_doc_embeddings_avg_val.npy',
        all_doc_embeddings_val.numpy())
np.save('save_embeddings/save_labels/all_labels_val.npy',
        np.array(all_labels_val))
print('Val shape:', all_doc_embeddings_val.shape)
