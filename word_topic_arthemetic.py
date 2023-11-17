import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
import sys


# Define the arithmetic operations on topic embeddings
def subtract_most_important(topic_embedding, word_embeddings, top_words, word_to_index):
    top_word_idx = word_to_index[top_words[0]]
    return topic_embedding - word_embeddings[:, top_word_idx]

def subtract_least_important(topic_embedding, word_embeddings, top_words, word_to_index):
    least_word_idx = word_to_index[top_words[-1]]
    return topic_embedding - word_embeddings[:, least_word_idx]

def add_most_subtract_least(topic_embedding, word_embeddings, top_words, word_to_index):
    top_word_idx = word_to_index[top_words[0]]
    least_word_idx = word_to_index[top_words[-1]]
    return topic_embedding + word_embeddings[:, top_word_idx] - word_embeddings[:, least_word_idx]

def subtract_most_add_least(topic_embedding, word_embeddings, top_words, word_to_index):
    top_word_idx = word_to_index[top_words[0]]
    least_word_idx = word_to_index[top_words[-1]]
    return topic_embedding - word_embeddings[:, top_word_idx] + word_embeddings[:, least_word_idx]

def add_most_important(topic_embedding, word_embeddings, top_words, word_to_index):
    top_word_idx = word_to_index[top_words[0]]
    return topic_embedding + word_embeddings[:, top_word_idx]

def add_least_important(topic_embedding, word_embeddings, top_words, word_to_index):
    least_word_idx = word_to_index[top_words[-1]]
    return topic_embedding + word_embeddings[:, least_word_idx]

def apply_operations_to_topic(topic_name, topic_embeddings, word_embeddings, vocab, operations):
    topic_idx = list(topic_labels.values()).index(topic_name)
    original_topic_distribution = np.dot(original_topic_embeddings[topic_idx], word_embeddings)
    top_word_idxs = np.argsort(original_topic_distribution)[-10:][::-1]
    original_top_words = [vocab[i] for i in top_word_idxs]
    
    print(f"\nOriginal {topic_name}: {' '.join(original_top_words)}")
    operation_results = [("Original", original_top_words, topic_embeddings[topic_idx])]

    for operation_name, operation_func in operations:
        modified_topic_embedding = operation_func(original_topic_embeddings[topic_idx], word_embeddings, original_top_words, word_to_index)
        modified_topic_distribution = np.dot(modified_topic_embedding, word_embeddings)
        top_word_idxs = np.argsort(modified_topic_distribution)[-10:][::-1]
        modified_top_words = [vocab[i] for i in top_word_idxs]
        
        print(f"{operation_name} {topic_name}: {' '.join(modified_top_words)}")
        operation_results.append((operation_name, modified_top_words, modified_topic_embedding))

    return operation_results

def visualize_operations_on_tsne(topic_name, topic_embeddings, word_embeddings, vocab, operations):
    operation_results = apply_operations_to_topic(topic_name, topic_embeddings, word_embeddings, vocab, operations)

    top_word_embeddings = []
    for _, words, _ in operation_results:
        for word in words:
            top_word_embeddings.append(word_embeddings[:, word_to_index[word]])
    top_word_embeddings = np.array(top_word_embeddings)

    combined_embeddings = np.vstack(([res[2] for res in operation_results], top_word_embeddings))

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results_combined = tsne.fit_transform(combined_embeddings)

    tsne_results_topics = tsne_results_combined[:len(operation_results)]
    tsne_results_words = tsne_results_combined[len(operation_results):]

    fig, ax = plt.subplots(figsize=(17, 17))

    texts = []

    for i, (label, words, _) in enumerate(operation_results):
        color = plt.cm.tab20(i)
        ax.scatter(tsne_results_topics[i, 0], tsne_results_topics[i, 1], color=color, label=label, s=100)
        texts.append(ax.text(tsne_results_topics[i, 0], tsne_results_topics[i, 1], label, fontsize=12, color=color))
        for j, word in enumerate(words):
            ax.scatter(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], color=color)
            texts.append(ax.text(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], word, fontsize=8, color=color))

    adjust_text(texts)
    ax.legend()
    plt.savefig('Automobiles.pdf', format='pdf')
    plt.show()

def main():
    path_to_embeddings = sys.argv[1]
    topic_name = sys.argv[2]

    topic_embeddings = np.load(f'{path_to_embeddings}/topic_embeddings.npy')
    word_embeddings = np.load(f'{path_to_embeddings}/word_embeddings.npy')
    vocab_values = np.load(f'{path_to_embeddings}/vocab_values.npy', allow_pickle=True)
    vocab_keys = np.load(f'{path_to_embeddings}/vocab_keys.npy', allow_pickle=True)
    vocab = {key: value for key, value in zip(vocab_keys, vocab_values)}

    global word_to_index, original_topic_embeddings
    word_to_index = {word: index for index, word in vocab.items()}
    original_topic_embeddings = np.copy(topic_embeddings)

    visualize_operations_on_tsne(topic_name, topic_embeddings, word_embeddings, vocab, operations)

if __name__ == '__main__':
    operations = [
        ("Subtract Most Important", subtract_most_important),
        ("Subtract Least Important", subtract_least_important),
        ("Add Most Important", add_most_important),
        ("Add Least Important", add_least_important),
        ("Add Most & Subtract Least", add_most_subtract_least),
        ("Subtract Most & Add Least", subtract_most_add_least)
    ]

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

    main()
