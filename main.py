import models
import data
import torch
from utils import TopicEval
from models.etmd_doc import ETMD
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from argparse import ArgumentParser


def main(settings):
    # precision
    torch.set_float32_matmul_precision('medium')
    # set seed
    seed_everything(42, workers=True)
    # import data
    train_loader, test_loader, val_loader, train_labels, test_labels, val_labels, text, vocab = data.get_data(
        settings['data_name'], settings['batch_size'])
    vocab_embeddings = None
    if settings['model_name'].startswith('etm'):
        vocab_embeddings = data.get_vocab_embeddings(vocab=vocab)
    # import model
    model = models.get_model(model_name=settings['model_name'],
                             vocab_size=len(vocab),
                             vocab_embeddings=vocab_embeddings,
                             topic_size=settings['topic_size'])
    # logger
    logger = TensorBoardLogger(
        settings['root_dir'], name=settings['model_name'], version=settings['data_name'])
    # train
    checkpoint_callback = ModelCheckpoint(monitor='val/loss',
                                          mode='min',
                                          save_top_k=1,
                                          filename='{epoch}-{val/loss:.2f}')
    trainer = Trainer(max_epochs=settings['max_epochs'],
                      callbacks=[checkpoint_callback],
                      accelerator=settings['accelerator'],
                      devices=settings['devices'],
                      default_root_dir=settings['root_dir'],
                      strategy=DDPStrategy(find_unused_parameters=False),
                      log_every_n_steps=1,
                      logger=logger)

    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # Save the embeddings and vocabulary after training
    model.save_embeddings(vocab, 'save_embeddings')
    torch.save(vocab, 'vocab/vocab.pt')

    # get topics and evaluate
    best_model_path = checkpoint_callback.best_model_path
    # Load the best model
    best_model = ETMD.load_from_checkpoint(checkpoint_path=best_model_path)
    print("Trainset", len(train_loader))
    print("Testset", len(test_loader))
    print("Validation", len(val_loader))
    # Generate and save document embeddings for the entire evaluation dataset
    best_model.generate_all_eval_doc_embeddings(
        train_loader, 'save_embeddings/train')
    best_model.generate_all_eval_doc_embeddings(
        val_loader, 'save_embeddings/validation')
    best_model.generate_all_eval_doc_embeddings(
        test_loader, 'save_embeddings/test')

    # Additional code for topic evaluations and logging remains the same ...
    topics = model.get_topics(vocab=vocab, path=best_model_path)
    eval = TopicEval(vocab=vocab, text=text)
    c_v = eval.topic_coherence(metric='c_v', topics=topics)
    c_npmi = eval.topic_coherence(metric='c_npmi', topics=topics)
    td = eval.topic_diversity(topics=topics)
    # save all results in .txt file
    logger.log_metrics({'c_v': c_v})
    logger.log_metrics({'c_npmi': c_npmi})
    logger.log_metrics({'td': td})
    path = trainer.log_dir + '/topics.txt'
    with open(path, 'w') as f:
        f.write('c_v: ' + str(c_v) + '\t' + 'c_npmi: ' +
                str(c_npmi) + '\t' + 'td: ' + str(td) + '\n')
        for i, topic in enumerate(topics):
            f.write('Topic ' + str(i+1) + ': ' + ' '.join(topic) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='20ng')
    parser.add_argument('--model_name', type=str, default='etm_dirichlet_rsvi')
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--root_dir', type=str, default='output/')
    parser.add_argument('--devices', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--topic_size', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=1)
    args = parser.parse_args()
    settings = vars(args)
    if settings['devices'] == -1:
        settings['devices'] = 'auto'
    main(settings)
