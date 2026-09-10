# Image clustering

Edit (Sep 2026): the notebook is now a plain Python script (`cluster.py`), the environment is managed with [uv](https://docs.astral.sh/uv/) instead of conda, and the script actually finishes the job: it writes the deduplicated set to `output/`, plus a plot of the cluster evaluation. Silhouette-optimal clustering alone turned out to both merge distinct stickers and split identical ones, so the final set starts from the cluster representatives and is completed with a perceptual-hash (imagehash) sweep that keeps exactly one image per visually-identical group.

Edit (Dec 2023): [clustimage](https://github.com/erdogant/clustimage) seems to provide a convenient interface around what I was trying to do, below.  Let's try that, for simplicity!

Playing around with various clustering libraries and preprocessing steps to group together similar-looking images.

## Usage

```bash
uv sync
uv run cluster.py
```

Point it at a directory of images with `--images-dir` (default `images/`), and tune the cluster-count search range with `--min-clust`/`--max-clust` (defaults 10–150, same as the original notebook). The deduplicated set lands in `output/`, alongside `clusteval.png` showing how the number of clusters was chosen.

I originally started this after [Google deprecated](https://9to5google.com/2021/09/28/gboard-minis-going-away-october/) their ["Minis" custom stickers](https://ai.googleblog.com/2017/05/neural-network-generated-illustrations.html). I used [Takeout](https://takeout.google.com) to download my Hangouts/Chat history, but there were multiple copies of each sticker that I used multiple times. Seeing as that I wanted to practice doing clustering, this was the perfect opportunity to put this to use.

| ![Example image](https://github.com/rtkfan/image-clusters/blob/master/example.png) |
| -- |
| Example Sweet Mini. I've sent hundreds of these. Don't judge me! |

The actual clustering is based largely on [this article](https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34). I'll run this with a variety of cluster counts, and see which one gives me the lowest k-means inertia. As separate instances of these images should be effectively identical, I expect the inertia to actually be 0 when I've identified the right number of clusters.

I'll use this to identify individual images, and then upload a deduplicated set somewhere I can continue to send these!
