import numpy as np

# Load the embeddings from the .npy files
train_embeddings = np.load(
    'save_embeddings/train/all_eval_doc_embeddings_train.npy')
test_embeddings = np.load(
    'save_embeddings/test/all_eval_doc_embeddings_test.npy')

# Concatenate along the first dimension (i.e., stacking them on top of each other)
all_embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)
print(all_embeddings.shape)

# Save the concatenated embeddings
np.save('save_embeddings/all_doc_embeddings.npy', all_embeddings)
