"""
Functions to capture embeddings at each step of training and save them as a movie
"""

import torch as t
import matplotlib.pyplot as plt
import os
from datetime import datetime

def run_movie_cmd():
    if not os.path.exists("movies"):
        os.mkdir("movies")
    mp4_name = os.path.join('movies', f'movie_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    os.system(f'ffmpeg -framerate 3 -i frames/embeddings_movie_%04d.png -c:v libx264 -pix_fmt yuv420p {mp4_name}')
    os.system('rm frames/embeddings_movie_*.png')
    print(f'movie saved as {mp4_name}')

def plot_embeddings_movie(model, step):
    plt.clf()
    embeddings = model.embedding.detach().cpu() # vocab_size x embed_dim
    chunked = t.chunk(embeddings, embeddings.shape[-1]//2, dim = -1)
    n = embeddings.shape[-1]//2
    # calculate number of rows and columns for subplots
    rows = int(n ** 0.5)
    cols = n // rows
    if rows * cols < n:  # if not enough subplots, add an extra column
        cols += 1
    # visualise each vocab_size x 2 chunk in a subplot
    _, axs = plt.subplots(rows, cols, figsize=(30, 15))
    axs = axs.flatten()  # flatten the array of axes to simplify indexing
    for i, chunk in enumerate(chunked):
        axs[i].scatter(chunk[:, 0], chunk[:, 1])
        words = [str(i) for i in range(embeddings.shape[0])]
        for j, word in enumerate(words):
            axs[i].annotate(word, xy=(chunk[j, 0], chunk[j, 1]))
    plt.tight_layout()  # adjust spacing between subplots
    # make /frames if it does not exist
    if not os.path.exists("frames"):
        os.mkdir("frames")
    plt.savefig(f"frames/embeddings_movie_{step:04}.png")  # change 'i' to your step variable
    plt.close()
