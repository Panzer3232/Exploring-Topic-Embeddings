import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl


class ETMD(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 vocab_embeddings,
                 topic_size,
                 beta=2.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.vocab_embeddings = vocab_embeddings
        self.topic_size = topic_size
        self.beta = beta
        self.embedding_size = vocab_embeddings.shape[1]

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.vocab_size, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Linear(in_features=100, out_features=self.topic_size),
        )
        self.encoder_norm = nn.BatchNorm1d(
            num_features=self.topic_size, eps=0.001, momentum=0.001, affine=True)
        self.encoder_norm.weight.data.copy_(torch.ones(self.topic_size))
        self.encoder_norm.weight.requires_grad = False

        # decoder
        self.topic_embeddings = nn.Linear(
            self.topic_size, self.embedding_size, bias=False)
        # print(self.topic_embeddings.shape)
        self.word_embeddings = nn.Linear(
            self.embedding_size, self.vocab_size, bias=False)
        # print(self.word_embeddings.shape)
        # initialize linear layer with pre-trained embeddings
        self.word_embeddings.weight.data.copy_(vocab_embeddings)
        self.decoder_norm = nn.BatchNorm1d(
            num_features=self.vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_norm.weight.data.copy_(torch.ones(self.vocab_size))
        self.decoder_norm.weight.requires_grad = False

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        alpha = F.softplus(self.encoder_norm(self.encoder(x)))
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        dist = Dirichlet(alpha)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        topic_embeddings = self.topic_embeddings(z)  # (batch_size, 300)
        word_embeddings = self.word_embeddings.weight  # (vocab_size, 300)
        # dot product
        # (batch_size, vocab_size)
        x_recon = torch.matmul(topic_embeddings, word_embeddings.T)
        x_recon = F.log_softmax(self.decoder_norm(
            x_recon), dim=1)  # (batch_size, vocab_size)
        return x_recon, dist

    def training_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, dist = self(x)
        recon, kl = self.objective(x, x_recon, dist)
        loss = recon + kl
        self.log_dict({'train/loss': loss,
                       'train/recon': recon,
                       'train/kl': kl},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['bow'].float()
        x_recon, dist = self(x)
        recon, kl = self.objective(x, x_recon, dist)
        loss = recon + kl
        self.log_dict({'val/loss': loss,
                       'val/recon': recon,
                       'val/kl': kl},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def objective(self, x, x_recon, dist):
        recon = -torch.sum(x * x_recon, dim=1).mean()
        prior = Dirichlet(torch.ones(self.topic_size, device=x.device) * 0.02)
        kl = self.beta * \
            torch.distributions.kl.kl_divergence(dist, prior).mean()
        return recon, kl

    def save_embeddings(self, vocab, path):
        vocab_id2word = {v: k for k, v in vocab.items()}
        # get embeddings
        topic_embeddings = self.topic_embeddings.weight.data.cpu().numpy().T  # (K, E)
        word_embeddings = self.word_embeddings.weight.data.cpu().numpy().T  # (E, V)
        vocab_embeddings = self.vocab_embeddings.cpu().numpy().T  # (V, E)
        print("Vocab Embeddings", vocab_embeddings.shape)
        print("Word Embeddings", word_embeddings.shape)

        topics = topic_embeddings @ word_embeddings  # (K, V)

        # save embeddings and vocabulary
        np.save(path + '/topic_embeddings.npy', topic_embeddings)
        np.save(path + '/word_embeddings.npy', word_embeddings)
        np.save(path + '/vocab_embeddings.npy',
                vocab_embeddings)  # Save vocab embeddings
        np.save(path + '/topics.npy', topics)
        np.save(path + '/vocab_keys.npy', np.array(list(vocab_id2word.keys())))
        np.save(path + '/vocab_values.npy',
                np.array(list(vocab_id2word.values())))

    def get_topics(self, vocab, path):
        # load best model
        model = self.load_from_checkpoint(path)
        print(path)
        model.eval()
        model.freeze()
        vocab_id2word = {v: k for k, v in vocab.items()}
       # print(vocab_id2word)
        # get topics
        topic_embeddings_shape = model.topic_embeddings.weight.shape
       # print("Shape of topic embeddings:", topic_embeddings_shape)
        topic_embeddings = model.topic_embeddings.weight.data.cpu().numpy().T  # (K, E)
        word_embeddings_shape = model.word_embeddings.weight.shape
       # print("Shape of word embeddings:", word_embeddings_shape)
        word_embeddings = model.word_embeddings.weight.data.cpu().numpy().T  # (E, V)
       # print(topic_embeddings.shape)
       # print(word_embeddings.shape)
        topics = topic_embeddings @ word_embeddings  # (K, V)
       # print(topics)
        topics = topics.argsort(axis=1)[:, ::-1]
        # top 10 words
        topics = topics[:, :10]
        topics = [[vocab_id2word[i] for i in topic] for topic in topics]

        return topics
