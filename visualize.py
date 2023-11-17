import sys
from adjustText import adjust_text
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from numpy.linalg import norm

# base_path is the path to embeddings

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def visualize(base_path):
    # Load your embeddings and vocab files
    topic_embeddings = np.load(base_path + 'topic_embeddings.npy')
    word_embeddings = np.load(base_path + 'word_embeddings.npy')
    vocab_values = np.load(base_path + 'vocab_values.npy', allow_pickle=True)
    vocab_keys = np.load(base_path + 'vocab_keys.npy', allow_pickle=True)

    # Create a lookup dictionary
    vocab = {key: value for key, value in zip(vocab_keys, vocab_values)}

    # Calculate the topic-word distributions
    topic_distributions = np.dot(topic_embeddings, word_embeddings)

    # Get top 10 words for each topic
    top_words = []
    top_word_idxs_list = []
    for topic_idx in range(topic_distributions.shape[0]):
        top_word_idxs = np.argsort(topic_distributions[topic_idx])[-10:][::-1]
        top_words_for_topic = [vocab[i] for i in top_word_idxs]
        top_words.append(top_words_for_topic)
        top_word_idxs_list.append(top_word_idxs)

    # Prepare embeddings for t-SNE
    top_word_embeddings = np.array(
        [word_embeddings[:, idx] for idxs in top_word_idxs_list for idx in idxs])
    
    

    # Combine topic and word embeddings
    combined_embeddings = np.vstack((topic_embeddings, top_word_embeddings))

    # Use t-SNE to visualize topics and words
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results_combined = tsne.fit_transform(combined_embeddings)

    # Split the t-SNE results back into topics and words
    tsne_results_topics = tsne_results_combined[:topic_embeddings.shape[0]]
    tsne_results_words = tsne_results_combined[topic_embeddings.shape[0]:]

    distinct_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a55FF', '#aaff00', '#ff00aa', '#000000', '#99ff99',
    '#6666ff', '#993399', '#ff9933', '#99ff33', '#33ff99',
    '#990000', '#003300', '#333300', '#003366', '#330066'
     ] 

    """topic_labels = {
    0: "US Poli-Elect",
    1: "Insur & Legal",
    2: "WebSec & Brows",
    3: "NFL & NBA",
    4: "Terror & Disast",
    5: "Global Conf",
    6: "Oil & Energy",
    7: "Isr-Pal Issue",
    8: "Econ & Forex",
    9: "Stock Trends",
    10: "Qtr Earnings",
    11: "Tech & Gadgets",
    12: "UN & Nukes",
    13: "NCAA Footbal",
    14: "Space & Sci",
    15: "Airline Crisis",
    16: "Golf & Tennis",
    17: "UEFA & F1",
    18: "MidEast Unrest",
    19: "Arts & Culture",
    20: "M&As",
    21: "MLB Stats",
    22: "Search & Media",
    23: "OS & Soft",
    24: "Mobile & 5G"
}"""
  
    """ topic_labels = {
    0: "Server Tech",
    1: "Market Trends",
    2: "Global Conflict",
    3: "India-Pak Poli",
    4: "US Politics",
    5: "Africa Crisis",
    6: "Consumer Tech",
    7: "Corp Takeovers",
    8: "Telecom",
    9: "Nuclear Policy",
    10: "Healthcare",
    11: "Corp Finance",
    12: "Space Explore",
    13: "Baseball",
    14: "Energy Market",
    15: "Golf & Soccer",
    16: "Digital Media",
    17: "Olympic Games",
    18: "Macroeconomics",
    19: "Basketball",
    20: "Web Privacy",
    21: "Terrorism",
    22: "Finance Crisis",
    23: "MidEast Policy",
    24: "American Soccer"
}"""
    """topic_labels = {
    0: "Mobile Tech",
    1: "Oil & Energy",
    2: "Space & Disaster",
    3: "Corp Earnings",
    4: "Computer Hardware",
    5: "Pharma & Financial Markets",
    6: "Sports & Injury",
    7: "Airline Labor",
    8: "Iraq Hostage",
    9: "Isr-Pal Relation",
    10: "US Elections",
    11: "OS & Web",
    12: "Golf & Tennis",
    13: "Euro Soccer",
    14: "Web & Tech",
    15: "Glob Economy",
    16: "Mergers",
    17: "Nuclear Policy",
    18: "US Baseball",
    19: "Iraq War",
    20: "Science Research",
    21: "South Asia Pol",
    22: "Legal Cases",
    23: "American Football",
    24: "African Conflict"
}"""
    topic_labels = {
        0: "ArmeniaHistory", 1: "Internet", 2: "Graphics Software", 3: "Christianity",
        4: "Encryption", 5: "Computer Hardware", 6: "Religion", 7: "Automobiles",
        8: "Ice Hockey", 9: "Space Mission", 10: "Incident", 11: "Product Sales",
        12: "Software Compilation", 13: "Sports", 14: "Violence", 15: "Random Letters",
        16: "Forum Discussions", 17: "Middle East Politics", 18: "Electronics", 
        19: "US Politics"
    }


    """topic_0_embedding = topic_embeddings[15]
    topic_1_embedding = topic_embeddings[20]
    similarity = cosine_similarity(topic_0_embedding, topic_1_embedding)
    print(f"Cosine similarity between topic 0 ({topic_labels[15]}) and topic 1 ({topic_labels[20]}): {similarity}")
    topic_0_embedding = topic_embeddings[5]
    topic_1_embedding = topic_embeddings[0]
    similarity = cosine_similarity(topic_0_embedding, topic_1_embedding)
    print(f"Cosine similarity between topic 0 ({topic_labels[5]}) and topic 1 ({topic_labels[0]}): {similarity}")
    topic_0_embedding = topic_embeddings[2]
    topic_1_embedding = topic_embeddings[24]
    similarity = cosine_similarity(topic_0_embedding, topic_1_embedding)
    print(f"Cosine similarity between topic 0 ({topic_labels[2]}) and topic 1 ({topic_labels[24]}): {similarity}")
    topic_0_embedding = topic_embeddings[2]
    topic_1_embedding = topic_embeddings[12]
    similarity = cosine_similarity(topic_0_embedding, topic_1_embedding)
    print(f"Cosine similarity between topic 0 ({topic_labels[2]}) and topic 1 ({topic_labels[12]}): {similarity}")
    topic_0_embedding = topic_embeddings[17]
    topic_1_embedding = topic_embeddings[5]
    similarity = cosine_similarity(topic_0_embedding, topic_1_embedding)
    print(f"Cosine similarity between topic 0 ({topic_labels[17]}) and topic 1 ({topic_labels[5]}): {similarity}")
    topic_0_embedding = topic_embeddings[13]
    topic_1_embedding = topic_embeddings[9]
    similarity = cosine_similarity(topic_0_embedding, topic_1_embedding)
    print(f"Cosine similarity between topic 0 ({topic_labels[13]}) and topic 1 ({topic_labels[9]}): {similarity}")
"""
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 10))

    # Annotate the topics and words
    texts = []

    for i in range(topic_distributions.shape[0]):
    # Print top 10 words for each topic
        print(f"{topic_labels[i]}: {' '.join(top_words[i])}")
        ax.scatter(tsne_results_topics[i, 0], tsne_results_topics[i, 1], color=distinct_colors[i], label=topic_labels[i], s=40)
        texts.append(ax.text(
            tsne_results_topics[i, 0], tsne_results_topics[i, 1], topic_labels[i], fontsize=10))

        for j in range(10):
            ax.scatter(tsne_results_words[i*10 + j, 0],
                   tsne_results_words[i*10 + j, 1], color=distinct_colors[i])
            ax.text(tsne_results_words[i*10 + j, 0],
                   tsne_results_words[i*10 + j, 1], top_words[i][j], fontsize=7)

    # Adjust the texts with adjust_text()
    adjust_text(texts)

    #ax.legend()
    #plt.savefig('ag_news.pdf', format='pdf')

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        print("Please provide the base path to embeddings as a command line argument.")
        sys.exit(1)

    visualize(base_path)


