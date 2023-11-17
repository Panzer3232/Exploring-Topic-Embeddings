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
        self.topic_probs = F.softmax(self.encoder_norm(
            self.encoder(x)), dim=1)  # Save the topic probabilities
        print(self.topic_probs.shape)
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

    # In your ETMD class

    def generate_all_eval_doc_embeddings(self, eval_dataloader, path):
        self.eval()  # Set the model to evaluation mode
        # Initialize an empty list to collect all document embeddings
        all_eval_doc_embeddings = []
        all_eval_labels = []  # Initialize an empty list to collect all true labels

        with torch.no_grad():  # Deactivate gradients for the following block
            for batch in eval_dataloader:
                x = batch['bow'].float()
                labels = batch['label']  # True labels from the current batch

                # Forward pass to get the reconstructions and distributions
                x_recon, dist = self(x)

            # Compute the document embeddings for this batch
                doc_embeddings = torch.matmul(
                    self.topic_probs, self.topic_embeddings.weight.T)

            # Collect the document embeddings and true labels
                all_eval_doc_embeddings.append(
                    doc_embeddings.cpu().detach().numpy())
                all_eval_labels.append(labels.cpu().detach().numpy())

        # Concatenate all the collected embeddings and labels
        all_doc_embeddings = np.concatenate(all_eval_doc_embeddings, axis=0)
        all_labels = np.concatenate(all_eval_labels, axis=0)

        print("all_doc shape:", all_doc_embeddings.shape)
        print("all_labels shape:", all_labels.shape)
        print("Labels", all_labels)

        # Save the embeddings and labels to disk
        np.save(f"{path}/all_eval_doc_embeddings.npy", all_doc_embeddings)
        np.save(f"{path}/all_eval_labels.npy", all_labels)

    def save_embeddings(self, vocab, path):
        vocab_id2word = {v: k for k, v in vocab.items()}
        # get embeddings
        topic_embeddings = self.topic_embeddings.weight.data.cpu().numpy().T  # (K, E)
        word_embeddings = self.word_embeddings.weight.data.cpu().numpy().T  # (E, V)
        topics = topic_embeddings @ word_embeddings  # (K, V)
        # save embeddings and vocabulary
        doc_embeddings = torch.matmul(
            self.topic_probs, self.topic_embeddings.weight.T)  # (batch_size, 300)
        # print("Doc0_embeddings", doc_embeddings.shape)
        np.save(path + '/doc_embeddings.npy',
                doc_embeddings.data.cpu().numpy())
        np.save(path + '/topic_embeddings.npy', topic_embeddings)
        np.save(path + '/word_embeddings.npy', word_embeddings)
        np.save(path + '/topics.npy', topics)
        np.save(path + '/vocab_keys.npy', np.array(list(vocab_id2word.keys())))
        np.save(path + '/vocab_values.npy',
                np.array(list(vocab_id2word.values())))

    def get_topics(self, vocab, path):
        # Load best model
        model = self.load_from_checkpoint(path)
        model.eval()
        model.freeze()
        vocab_id2word = {v: k for k, v in vocab.items()}

        # Get topics
        topic_embeddings = model.topic_embeddings.weight.data.cpu().numpy().T  # (K, E)
        word_embeddings = model.word_embeddings.weight.data.cpu().numpy().T  # (E, V)

        # Load document embeddings and get top topics
        doc_embeddings = np.load(
            'save_embeddings/doc_embeddings.npy')  # (N, E)
        # print("Doc_embeddings", doc_embeddings.shape)
        doc_topics = doc_embeddings @ topic_embeddings.T  # (N, K)
        # print("Doc_topic", doc_topics.shape)
        doc_topics = doc_topics.argsort(axis=1)[:, ::-1]  # (N, K)
        doc_topics = doc_topics[:, :5]  # (N, 5)

        # Get top words for top topics
        top_topics = topic_embeddings[doc_topics, :]  # (N, 5, E)
        top_topic_words = top_topics @ word_embeddings  # (N, 5, V)
        top_topic_words = top_topic_words.argsort(
            axis=2)[:, :, ::-1]  # (N, 5, V)
        top_topic_words = top_topic_words[:, :, :10]  # (N, 5, 10)

        # Convert word indices to words
        top_topic_words = [vocab_id2word[i]
                           for doc_topic_word in top_topic_words for topic_word in doc_topic_word for i in topic_word]

        return top_topic_words
