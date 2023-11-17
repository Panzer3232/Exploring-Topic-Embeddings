# Exploring Topic Embeddings



## Exploring Topic Embeddings Visualization Script
This script provides a method to visualize the relationship between topics and their top words. It uses t-SNE, a technique for dimensionality reduction, to map high-dimensional data to a two-dimensional plane. The script loads embeddings and vocabulary from specified files, computes topic-word distributions, and identifies top words for each topic. It then applies t-SNE to these embeddings and plots the resulting two-dimensional points.

## Requirements

To run the script, you will need the following Python libraries: numpy, matplotlib, scikit-learn, adjustText

<code> pip install numpy matplotlib scikit-learn adjustText</code>

## Usage
You can run the script using Python from your command line like this:

<code>python visualize.py [path_to_embeddings]
</code>
