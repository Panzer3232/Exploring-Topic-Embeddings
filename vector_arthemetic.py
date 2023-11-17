import sys
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.metrics.pairwise import cosine_similarity

def load_data(embeddings_path):
    topic_embeddings = np.load(f'{embeddings_path}/topic_embeddings.npy')
    word_embeddings = np.load(f'{embeddings_path}/word_embeddings.npy')
    vocab_keys = np.load(f'{embeddings_path}/vocab_keys.npy', allow_pickle=True)
    vocab_values = np.load(f'{embeddings_path}/vocab_values.npy', allow_pickle=True)
    return topic_embeddings, word_embeddings, vocab_keys, vocab_values

def create_vocab(vocab_keys, vocab_values):
    return {key: value for key, value in zip(vocab_keys, vocab_values)}

def compute_compound_embedding(sequential_operations, topic_labels, topic_embeddings, word_embeddings, vocab):
    reversed_vocab = {v: k for k, v in vocab.items()}
    initial_name = sequential_operations[0]
    
    if initial_name in topic_labels.values():
        initial_idx = list(topic_labels.values()).index(initial_name)
        compound_embedding = np.copy(topic_embeddings[initial_idx])
    elif initial_name in reversed_vocab:
        word_idx = reversed_vocab[initial_name]
        compound_embedding = np.copy(word_embeddings[:, word_idx])
    else:
        print(f"Invalid initial name: {initial_name}")
        sys.exit(1)

    for i in range(1, len(sequential_operations)):
        operator_and_name = sequential_operations[i]
        operator = operator_and_name[0]
        name = operator_and_name[1:]

        if name in topic_labels.values():
            topic_idx = list(topic_labels.values()).index(name)
            embedding = topic_embeddings[topic_idx]
        elif name in reversed_vocab:
            word_idx = reversed_vocab[name]
            embedding = word_embeddings[:, word_idx]
        else:
            print(f"Invalid name: {name}")
            sys.exit(1)

        if operator == "+":
            compound_embedding += embedding
        elif operator == "-":
            compound_embedding -= embedding
        else:
            print(f"Invalid operator: {operator}")
            sys.exit(1)

    return compound_embedding

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
        ax.scatter(tsne_results_topics[i, 0], tsne_results_topics[i, 1], color=color, s=100)
        texts.append(ax.text(tsne_results_topics[i, 0], tsne_results_topics[i, 1], extended_topic_labels[i], fontsize=12))
        word_color = color if color == 'black' else plt.cm.tab20(i)
        for j, word in enumerate(words):
            ax.scatter(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], color=word_color,s=60)
            ax.text(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], word, fontsize=9)
    adjust_text(texts)
    plt.savefig('Arthemetic.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vector_arithmetic.py <path_to_embeddings> <operation_string>")
        sys.exit(1)

    embeddings_path = sys.argv[1]
    operation_string = sys.argv[2]

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
    reversed_vocab = {v: k for k, v in vocab.items()}

    sequential_operations = [op.strip() for op in operation_string.split(',')]
    compound_embedding = compute_compound_embedding(sequential_operations, topic_labels, topic_embeddings, word_embeddings, vocab)

    initial_name = sequential_operations[0]
    if initial_name in topic_labels.values():
        initial_idx = list(topic_labels.values()).index(initial_name)
        initial_embedding = np.copy(topic_embeddings[initial_idx])
    elif initial_name in reversed_vocab:
        word_idx = reversed_vocab[initial_name]
        initial_embedding = np.copy(word_embeddings[:, word_idx])
    else:
        print(f"Invalid initial name: {initial_name}")
        sys.exit(1)

    similarity = cosine_similarity(initial_embedding.reshape(1, -1), compound_embedding.reshape(1, -1))
    print(f"Cosine similarity between the first topic '{initial_name}' and the resultant topic is: {similarity[0][0]}")

    extended_topic_embeddings = np.vstack((topic_embeddings, compound_embedding))
    topic_distributions = np.dot(extended_topic_embeddings, word_embeddings)
    top_word_indices = [np.argsort(dist)[-10:][::-1] for dist in topic_distributions]
    top_words = [[vocab[idx] for idx in indices] for indices in top_word_indices]
    extended_topic_labels = {**topic_labels, topic_embeddings.shape[0]: "New_Topic"}
    visualize_topics_and_words(extended_topic_embeddings, word_embeddings, top_words, extended_topic_labels)
