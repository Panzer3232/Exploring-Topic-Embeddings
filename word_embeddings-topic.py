import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text


def load_data(embeddings_path):
    topic_embeddings = np.load(f'{embeddings_path}/topic_embeddings.npy')
    word_embeddings = np.load(f'{embeddings_path}/word_embeddings.npy')
    vocab_keys = np.load(f'{embeddings_path}/vocab_keys.npy', allow_pickle=True)
    vocab_values = np.load(f'{embeddings_path}/vocab_values.npy', allow_pickle=True)
    return topic_embeddings, word_embeddings, vocab_keys, vocab_values

def create_vocab(vocab_keys, vocab_values):
    return {key: value for key, value in zip(vocab_keys, vocab_values)}

"""def compute_compound_topic(topic_name, word, operation, topic_labels, topic_embeddings, word_embeddings, vocab):
    topic_idx = list(topic_labels.values()).index(topic_name)
    
    # Reverse the vocab dictionary to go from word to index
    reversed_vocab = {v: k for k, v in vocab.items()}
    
    if word not in reversed_vocab:
        raise ValueError(f"The word {word} is not in the vocabulary.")
        
    word_idx = reversed_vocab[word]  # Get the index of the word
    
    if operation == "add":
        return topic_embeddings[topic_idx] + word_embeddings[:, word_idx]
    elif operation == "subtract":
        return topic_embeddings[topic_idx] - word_embeddings[:, word_idx]
    else:
        raise ValueError(f"Unsupported operation: {operation}")"""

def compute_compound_topic(topic_name, operation_sequence, topic_labels, topic_embeddings, word_embeddings, vocab):
    topic_idx = list(topic_labels.values()).index(topic_name)
    new_topic_embedding = np.copy(topic_embeddings[topic_idx])

    reversed_vocab = {v: k for k, v in vocab.items()}

    operations = operation_sequence.split(",")
    for op in operations:
        action = op[0]
        word = op[1:]
        
        if word not in reversed_vocab:
            raise ValueError(f"The word {word} is not in the vocabulary.")
        
        word_idx = reversed_vocab[word]
        
        if action == '+':
            new_topic_embedding += word_embeddings[:, word_idx]
        elif action == '-':
            new_topic_embedding -= word_embeddings[:, word_idx]
        else:
            raise ValueError(f"Unsupported operation: {action}")

    return new_topic_embedding





def visualize_topics_and_words(extended_topic_embeddings, word_embeddings, top_words, extended_topic_labels):
    top_word_embeddings = np.array([word_embeddings[:, idx] for indices in top_word_indices for idx in indices])
    combined_embeddings = np.vstack((extended_topic_embeddings, top_word_embeddings))
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results_combined = tsne.fit_transform(combined_embeddings)
    tsne_results_topics = tsne_results_combined[:extended_topic_embeddings.shape[0]]
    tsne_results_words = tsne_results_combined[extended_topic_embeddings.shape[0]:]

    fig, ax = plt.subplots(figsize=(20, 20))
    texts = []
    for i, words in enumerate(top_words):
        print(f"{extended_topic_labels[i]}: {' '.join(words)}")
        if i == topic_embeddings.shape[0]:
            color = 'black'
        else:
            color = plt.cm.tab20(i)
        ax.scatter(tsne_results_topics[i, 0], tsne_results_topics[i, 1], color=color, s=50)
        texts.append(ax.text(tsne_results_topics[i, 0], tsne_results_topics[i, 1], extended_topic_labels[i], fontsize=10))
        word_color = color if color == 'black' else plt.cm.tab20(i)
        for j, word in enumerate(words):
            ax.scatter(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], color=word_color)
            ax.text(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], word, fontsize=7)
    adjust_text(texts)
    ax.legend()
    plt.savefig('Sports+nhl-pitching.pdf', format='pdf')
    #print(tsne_results_topics.min(axis=0), tsne_results_topics.max(axis=0))
    #print(tsne_res_words.min(axis=0), tsne_results_words.max(axis=0))
    #ax.set_xlim([-15, 13])
    #ax.set_ylim([-14, 11])
  
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <embeddings_path> <topic_name> <operation_sequence>")
        sys.exit(1)

    embeddings_path = sys.argv[1]
    topic_name = sys.argv[2]
    operation_sequence = sys.argv[3]

    # Replace this dictionary with your real topic labels
    topic_labels = {
        0: "ArmeniaHistory",
        1: "Internet",
        2: "Graphics Software",
        3: "Christianity",
        4: "Encryption",
        5: "Computer Hardware",
        6: "Religion",
        7: "Automobiles",
        8: "Ice Hockey",
        9: "Space Mission",
        10: "Incident",
        11: "Product Sales",
        12: "Software Compilation",
        13: "Sports",
        14: "Violence",
        15: "Random Letters",
        16: "Forum Discussions",
        17: "Middle East Politics",
        18: "Electronics",
        19: "US Politics"
    }

    topic_embeddings, word_embeddings, vocab_keys, vocab_values = load_data(embeddings_path)
    vocab = create_vocab(vocab_keys, vocab_values)
    new_topic_embedding = compute_compound_topic(topic_name, operation_sequence, topic_labels, topic_embeddings, word_embeddings, vocab)
    
    """ extended_topic_embeddings = np.vstack((topic_embeddings, new_topic_embedding.reshape(1, -1)))
    extended_topic_labels = list(topic_labels.values()) + ["New_Topic"]

    topic_distributions = np.dot(extended_topic_embeddings, word_embeddings)
    top_word_indices = np.argsort(topic_distributions, axis=0)[-10:][::-1]
    top_words = [[vocab[idx] for idx in indices] for indices in top_word_indices.T]
    
    visualize_topics_and_words(extended_topic_embeddings, top_words, extended_topic_labels)"""

    extended_topic_embeddings = np.vstack((topic_embeddings,new_topic_embedding))
    topic_distributions = np.dot(extended_topic_embeddings, word_embeddings)
    top_word_indices = [np.argsort(dist)[-10:][::-1] for dist in topic_distributions]
    top_words = [[vocab[idx] for idx in indices] for indices in top_word_indices]
    extended_topic_labels = {**topic_labels, topic_embeddings.shape[0]: "New_Topic"}
    visualize_topics_and_words(extended_topic_embeddings, word_embeddings, top_words, extended_topic_labels)
