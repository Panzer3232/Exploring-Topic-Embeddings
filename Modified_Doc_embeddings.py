
import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

def load_data(base_path):
    doc_embeddings = np.load(base_path + 'doc_embeddings.npy')
    topic_embeddings = np.load(base_path + 'topic_embeddings.npy')
    word_embeddings = np.load(base_path + 'word_embeddings.npy')
    vocab_values = np.load(base_path + 'vocab_values.npy', allow_pickle=True)
    vocab_keys = np.load(base_path + 'vocab_keys.npy', allow_pickle=True)
    vocab = {key: value for key, value in zip(vocab_keys, vocab_values)}
    return doc_embeddings, topic_embeddings, word_embeddings, vocab

def get_top_words(topic_embeddings, word_embeddings, vocab):
    topic_distributions = np.dot(topic_embeddings, word_embeddings)
    top_words = []
    top_word_idxs_list = []
    for topic_idx in range(topic_distributions.shape[0]):
        top_word_idxs = np.argsort(topic_distributions[topic_idx])[-10:][::-1]
        top_words_for_topic = [vocab[i] for i in top_word_idxs]
        top_words.append(top_words_for_topic)
        top_word_idxs_list.append(top_word_idxs)
    return top_words, top_word_idxs_list

def corrected_visualize_v3(base_path):
    doc_embeddings, topic_embeddings, word_embeddings, vocab = load_data(base_path)
    top_words, top_word_idxs_list = get_top_words(topic_embeddings, word_embeddings, vocab)
    top_word_embeddings = np.array([word_embeddings[:, idx] for idxs in top_word_idxs_list for idx in idxs])
    combined_embeddings = np.vstack((doc_embeddings, topic_embeddings, top_word_embeddings))
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results_combined = tsne.fit_transform(combined_embeddings)
    tsne_results_docs = tsne_results_combined[:doc_embeddings.shape[0]]
    tsne_results_topics = tsne_results_combined[doc_embeddings.shape[0]:doc_embeddings.shape[0]+topic_embeddings.shape[0]]
    tsne_results_words = tsne_results_combined[doc_embeddings.shape[0]+topic_embeddings.shape[0]:]
    topic_labels = {
        idx: label for idx, label in enumerate(np.load(base_path + 'topics.npy'))
    }
    fig, ax = plt.subplots(figsize=(20, 20))
    texts = []
    ax.scatter(tsne_results_docs[:, 0], tsne_results_docs[:, 1], color='black', s=50, alpha=0.5, marker='o')
    for i in range(len(topic_labels)):
        ax.scatter(tsne_results_topics[i, 0], tsne_results_topics[i, 1], color=plt.cm.tab20(i), s=100, marker='o')
        texts.append(ax.text(tsne_results_topics[i, 0], tsne_results_topics[i, 1], topic_labels[i], fontsize=12))
        for j in range(10):
            ax.scatter(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], color=plt.cm.tab20(i), s=20, marker='o')
            texts.append(ax.text(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], top_words[i][j], fontsize=8))
    adjust_text(texts)
    ax.legend()
    plt.savefig('Modified_Doc_Embeddings.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
        corrected_visualize_v3(base_path)
    else:
        print("Please provide the base path to embeddings as a command line argument.")
