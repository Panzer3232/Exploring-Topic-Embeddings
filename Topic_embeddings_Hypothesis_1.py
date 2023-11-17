#!/usr/bin/env python
# coding: utf-8

from adjustText import adjust_text
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load your embeddings and vocab files
topic_embeddings = np.load('./new-data/save_embeddings/topic_embeddings.npy')
word_embeddings = np.load('./new-data/save_embeddings/word_embeddings.npy')
vocab_values = np.load('./new-data/save_embeddings/vocab_values.npy', allow_pickle=True)
vocab_keys = np.load('./new-data/save_embeddings/vocab_keys.npy', allow_pickle=True)

# Create a lookup dictionary
vocab = {key: value for key, value in zip(vocab_keys, vocab_values)}

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

# Index for Christianity and Religion topics
topic1 = list(topic_labels.keys())[list(topic_labels.values()).index("Middle East Politics")]
topic2 = list(topic_labels.keys())[list(topic_labels.values()).index("US Politics")]
topic3 = list(topic_labels.keys())[list(topic_labels.values()).index("Religion")]

# Subtract Christianity embedding with Religion embedding
new_topic_embedding = (topic_embeddings[topic1] - topic_embeddings[topic2]) + topic_embeddings[topic3]

# Add the new topic to the original topic embeddings
extended_topic_embeddings = np.vstack((topic_embeddings, new_topic_embedding))

# Calculate the topic-word distributions
topic_distributions = np.dot(extended_topic_embeddings, word_embeddings)

# Get top 10 words for each topic
top_words = []
top_word_idxs_list = []
for topic_idx in range(topic_distributions.shape[0]):
    top_word_idxs = np.argsort(topic_distributions[topic_idx])[-10:][::-1]
    top_words_for_topic = [vocab[i] for i in top_word_idxs]
    top_words.append(top_words_for_topic)
    top_word_idxs_list.append(top_word_idxs)

# Prepare embeddings for t-SNE
top_word_embeddings = np.array([word_embeddings[:, idx] for idxs in top_word_idxs_list for idx in idxs])

# Combine topic and word embeddings
combined_embeddings = np.vstack((extended_topic_embeddings, top_word_embeddings))

# Use t-SNE to visualize topics and words
tsne = TSNE(n_components=2, random_state=0)
tsne_results_combined = tsne.fit_transform(combined_embeddings)

# Split the t-SNE results back into topics and words
tsne_results_topics = tsne_results_combined[:extended_topic_embeddings.shape[0]]
tsne_results_words = tsne_results_combined[extended_topic_embeddings.shape[0]:]

# Extend the topic labels with the new topic
extended_topic_labels = {**topic_labels, topic_embeddings.shape[0]: "New_Topic"}

# Visualization
fig, ax = plt.subplots(figsize=(17, 17))

# Annotate the topics and words
texts = []
for i in range(topic_distributions.shape[0]):
    print(f"{extended_topic_labels[i]}: {' '.join(top_words[i])}")  # Print top 10 words for each topic
    ax.scatter(tsne_results_topics[i, 0], tsne_results_topics[i, 1], color=plt.cm.tab20(i), label=extended_topic_labels[i], s=100)
    texts.append(ax.text(tsne_results_topics[i, 0], tsne_results_topics[i, 1], extended_topic_labels[i], fontsize=15))

    for j in range(10):
        ax.scatter(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], color=plt.cm.tab20(i))
        ax.text(tsne_results_words[i*10 + j, 0], tsne_results_words[i*10 + j, 1], top_words[i][j], fontsize=10)

# Adjust the texts with adjust_text()
adjust_text(texts)

ax.legend()

# Save the figure as a PDF
#plt.savefig('extended_topic_visualization.pdf', format='pdf')

plt.show()

def print_top_words(topic1, topic2, topic3):
    # Find the indices for the given topics
    topic1_idx = list(topic_labels.keys())[list(topic_labels.values()).index(topic1)]
    topic2_idx = list(topic_labels.keys())[list(topic_labels.values()).index(topic2)]
    topic3_idx = list(topic_labels.keys())[list(topic_labels.values()).index(topic3)]

    # Compute new topic
    new_topic = (topic_embeddings[topic1_idx] - topic_embeddings[topic2_idx]) + topic_embeddings[topic3_idx]

    # Add the new topic to the original topic embeddings
    extended_topic_embeddings = np.vstack((topic_embeddings, new_topic))

    # Calculate the topic-word distributions
    topic_distributions = np.dot(extended_topic_embeddings, word_embeddings)

    # Get top 10 words for the new topic
    top_word_idxs = np.argsort(topic_distributions[-1])[-10:][::-1]
    top_words_for_topic = [vocab[i] for i in top_word_idxs]

    # Print top 10 words for the new topic
    print(f"New topic (based on {topic1}-{topic2}+{topic3}): {' '.join(top_words_for_topic)}")

# Test with some topics
print_top_words("Computer Hardware", "Graphics Software", "Encryption")
print_top_words("Graphics Software", "Computer Hardware", "Internet")
print_top_words("Middle East Politics", "US Politics", "Religion")
print_top_words("Product Sales", "Automobiles", "Internet")
print_top_words("Sports", "Ice Hockey", "Electronics")
print_top_words("Christianity", "US Politics", "Internet")

# Index for Computer Hardware, Graphics Software, and Encryption topics
topic1 = list(topic_labels.keys())[list(topic_labels.values()).index("Computer Hardware")]
topic2 = list(topic_labels.keys())[list(topic_labels.values()).index("Graphics Software")]
topic3 = list(topic_labels.keys())[list(topic_labels.values()).index("Encryption")]

# Compute HardwareSecurity topic
hardware_security = (topic_embeddings[topic1] - topic_embeddings[topic2]) + topic_embeddings[topic3]

# Add the new topic to the original topic embeddings
extended_topic_embeddings = np.vstack((topic_embeddings, hardware_security))

# Calculate the topic-word distributions
topic_distributions = np.dot(extended_topic_embeddings, word_embeddings)

# Get top 10 words for the HardwareSecurity topic
top_word_idxs = np.argsort(topic_distributions[-1])[-10:][::-1]
top_words_for_topic = [vocab[i] for i in top_word_idxs]

# Print top 10 words for the HardwareSecurity topic
print(f"HardwareSecurity: {' '.join(top_words_for_topic)}")  # Print top 10 words for the new topic

# Now, reverse the equation
# Subtract Encryption embedding from HardwareSecurity and add Graphics Software embedding
reversed_computer_hardware = (hardware_security - topic_embeddings[topic3]) + topic_embeddings[topic2]

# Add the new reversed topic to the extended topic embeddings
extended_topic_embeddings = np.vstack((extended_topic_embeddings, reversed_computer_hardware))

# Calculate the topic-word distributions again
topic_distributions = np.dot(extended_topic_embeddings, word_embeddings)

# Get top 10 words for the ReversedComputerHardware topic
top_word_idxs = np.argsort(topic_distributions[-1])[-10:][::-1]
top_words_for_topic = [vocab[i] for i in top_word_idxs]

# Print top 10 words for the ReversedComputerHardware topic
print(f"ReversedComputerHardware: {' '.join(top_words_for_topic)}")  # Print top 10 words for the new topic
