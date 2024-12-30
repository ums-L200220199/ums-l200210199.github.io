from itertools import islice

def top_words(num_clusters, clusters, mtx, columns):
    import numpy as np
    top = []
    for i in range(num_clusters):  # A
        rows_in_cluster = np.where(clusters == i)[0]  # B
        word_freqs = mtx[rows_in_cluster].sum(axis=0).A[0]  # C
        ordered_freqs = np.argsort(word_freqs)  # D
        top_words = [
            (columns[idx], int(word_freqs[idx]))
            for idx in islice(reversed(ordered_freqs), 20)
        ]  # D
        top.append(top_words)
    return top
