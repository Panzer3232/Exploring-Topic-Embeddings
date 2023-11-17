import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader, Dataset
from .TokenEase import Pipe  # Assuming Pipe is in the same directory
import gensim.downloader as gensim_api


class TorchDatasetBoW(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'bow': self.data[idx]}


def get_vocab_embeddings(vocab: dict):
    model = gensim_api.load('glove-wiki-gigaword-300')
    embeddings = torch.zeros(len(vocab), 300)
    for i, word in enumerate(vocab):
        if word in model:
            embeddings[i] = torch.from_numpy(model[word].copy())
    return embeddings


def get_data(data_name, batch_size):
    if data_name == 'agnews':
        train_iterator, test_iterator = AG_NEWS(
            root='.data', split=('train', 'test'))
        train_data = [(label, text) for (label, text) in train_iterator]
        # first_few = [item for _, item in zip(range(5), train_iterator)]
        # print(first_few)
        test_data = [(label, text) for (label, text) in test_iterator]

        train_labels, train_text = zip(*train_data)
        test_labels, test_text = zip(*test_data)
        val_text, val_labels = test_text, test_labels

    else:
        raise NotImplementedError

    print("Sample train text:", train_text[:5])
    print("Type of train text:", type(train_text))
    print("Type of a sample element:", type(train_text[0]))

    pipe = Pipe(preprocess=True,
                max_df=0.80,
                min_df=50,
                doc_start_token='<s>',
                doc_end_token='</s>',
                unk_token='<unk>',
                email_token='<email>',
                url_token='<url>',
                number_token='<number>',
                alpha_num_token='<alpha_num>')

    train_bow = pipe.process_data(train_text)
    test_bow, _ = pipe.get_doc_bow(test_text)
    val_bow = test_bow
    text = pipe.text
    vocab = pipe.vocab

    train_dataset = TorchDatasetBoW(train_bow)
    test_dataset = TorchDatasetBoW(test_bow)
    val_dataset = TorchDatasetBoW(val_bow)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, val_loader, text, vocab


if __name__ == '__main__':
    train_loader, test_loader, val_loader, text, vocab = get_data('agnews', 32)
    print(text[0])
