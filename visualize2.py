import sys
import ast
from adjustText import adjust_text
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from numpy.linalg import norm
# defining cosine similarity

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

# function to get the average embeddings of the list of words

# def compute_word_average_embedding(words, vocab, word_embeddings):
#     indices = [vocab[word] for word in words if word in vocab]
#     embeddings = word_embeddings[:, indices]
#     avg_embedding = np.mean(embeddings, axis=1)
#     return avg_embedding

def compute_word_average_embedding(words, reversed_vocab, vocab_embeddings):
    resultant_embedding = np.zeros_like(vocab_embeddings[:, 0])  # Initialize with zero vector
    for word in words:
        if word in reversed_vocab:
            word_idx = reversed_vocab[word]
            word_embedding = vocab_embeddings[:, word_idx]
            resultant_embedding += word_embedding  # Sequential addition

    if np.any(resultant_embedding):  # To ensure that at least one valid word was provided
        #print(words)
        resultant_embedding /= len(words)  # Divide by the number of words

    return resultant_embedding

# base_path is the path to embeddings

def visualize(base_path, words_to_check, original_topic_index):
    # Load your embeddings and vocab files
    topic_embeddings = np.load(base_path + 'topic_embeddings.npy')
    word_embeddings = np.load(base_path + 'word_embeddings.npy')
    vocab_embeddings= np.load(base_path + 'vocab_embeddings.npy')
    vocab_values = np.load(base_path + 'vocab_values.npy', allow_pickle=True)
    vocab_keys = np.load(base_path + 'vocab_keys.npy', allow_pickle=True)
    
    # Create a lookup dictionary
    vocab = {key: value for key, value in zip(vocab_keys, vocab_values)}
    
    # Reverse the vocab dictionary to go from word to index
    reversed_vocab = {v: k for k, v in vocab.items()}
    
    # Calculate the topic-word distributions
    topic_distributions = np.dot(topic_embeddings, word_embeddings)
    
    avg_embedding = compute_word_average_embedding(words_to_check, reversed_vocab, vocab_embeddings)
    # Assuming that 15 is the index for "Random Letters" in topic_labels
    original_topic_embedding = topic_embeddings[original_topic_index, :]

    # Calculate cosine similarity between random_letters_topic_embedding and avg_embedding
    similarity = cosine_similarity(original_topic_embedding, avg_embedding)
    print(f"Cosine similarity between the topic embedding of RL and average word embedding: {similarity}")

    
    # Extend the topic_embeddings with the avg_embedding
    extended_topic_embeddings = np.vstack((topic_embeddings, avg_embedding))
    
    # Get top 10 words for the average embedding
    avg_embedding_distributions = np.dot(avg_embedding, word_embeddings)
    avg_top_word_idxs = np.argsort(avg_embedding_distributions)[-10:][::-1]
    avg_top_words = [vocab[i] for i in avg_top_word_idxs]
    
    # Get top 10 words for each topic
    top_words = []
    top_word_idxs_list = []
    for topic_idx in range(topic_distributions.shape[0]):
        top_word_idxs = np.argsort(topic_distributions[topic_idx])[-10:][::-1]
        top_words_for_topic = [vocab[i] for i in top_word_idxs]
        top_words.append(top_words_for_topic)
        top_word_idxs_list.append(top_word_idxs)

    top_word_idxs_list.append(avg_top_word_idxs)
    top_words.append(avg_top_words)

    # Prepare embeddings for t-SNE
    top_word_embeddings = np.array(
        [word_embeddings[:, idx] for idxs in top_word_idxs_list for idx in idxs])
    
    # Combine topic and word embeddings
    combined_embeddings = np.vstack((extended_topic_embeddings, top_word_embeddings))
    
    # Use t-SNE to visualize topics and words
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results_combined = tsne.fit_transform(combined_embeddings)

    # Split the t-SNE results back into topics and words
    tsne_results_topics = tsne_results_combined[:extended_topic_embeddings.shape[0]]
    tsne_results_words = tsne_results_combined[extended_topic_embeddings.shape[0]:]
    
    # Initialize topic labels
    topic_labels = {
        0: "ArmeniaHistory", 1: "Internet", 2: "Graphics Software", 3: "Christianity",
        4: "Encryption", 5: "Computer Hardware", 6: "Religion", 7: "Automobiles",
        8: "Ice Hockey", 9: "Space Mission", 10: "Incident", 11: "Product Sales",
        12: "Software Compilation", 13: "Sports", 14: "Violence", 15: "Random Letters",
        16: "Forum Discussions", 17: "Middle East Politics", 18: "Electronics", 
        19: "US Politics", 20: "New Topic"
    }

    # Visualization
    fig, ax = plt.subplots(figsize=(25, 25))
    
    # Annotate the topics and words
    texts = []

    for i in range(extended_topic_embeddings.shape[0]):
        print(f"{topic_labels[i]}: {' '.join(top_words[i])}")
        ax.scatter(tsne_results_topics[i, 0], tsne_results_topics[i, 1], color=plt.cm.tab20(
            i % 20), label=topic_labels[i], s=100)
        texts.append(ax.text(
            tsne_results_topics[i, 0], tsne_results_topics[i, 1], topic_labels[i], fontsize=11))

        for j in range(10):
            ax.scatter(tsne_results_words[i*10 + j, 0],
                       tsne_results_words[i*10 + j, 1], color=plt.cm.tab20(i % 20))
            ax.text(tsne_results_words[i*10 + j, 0],
                    tsne_results_words[i*10 + j, 1], top_words[i][j], fontsize=8)
    
    # Adjust the texts with adjust_text()
    adjust_text(texts)
    
    #ax.legend()
    
    plt.show()

    

if __name__ == "__main__":
    if len(sys.argv) > 3:
        base_path = sys.argv[1]
        words_to_check = ast.literal_eval(sys.argv[2])  # Assuming the words are provided as a list
        original_topic_index = int(sys.argv[3])  # Convert the argument to integer
    else:
        print("Please provide the base path to embeddings, a word list, and the index of the original topic as command-line arguments.")
        sys.exit(1)

    visualize(base_path, words_to_check, original_topic_index)


