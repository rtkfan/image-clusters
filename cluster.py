"""Cluster similar-looking images and extract a deduplicated set.

Loads every image in a directory, clusters them by visual similarity
(clustimage: PCA features, t-SNE embedding, agglomerative clustering with the
number of clusters chosen by silhouette score), saves the cluster-evaluation
plot, and writes a deduplicated set to the output directory: the cluster
representatives, plus any image the clustering dropped, minus perceptual
duplicates of what is already kept.
"""

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import imagehash
import numpy as np
from clustimage import Clustimage
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Directory of images to cluster (default: images)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for the deduplicated set and plot (default: output)",
    )
    parser.add_argument(
        "--min-clust",
        type=int,
        default=10,
        help="Minimum number of clusters to evaluate (default: 10)",
    )
    parser.add_argument(
        "--max-clust",
        type=int,
        default=150,
        help="Maximum number of clusters to evaluate (default: 150)",
    )
    parser.add_argument(
        "--phash-cutoff",
        type=int,
        default=0,
        help=(
            "Perceptual-hash hamming distance below which two images count as "
            "duplicates (default: 0, i.e. identical hashes only)"
        ),
    )
    return parser.parse_args()


def representative_pathnames(cl, results):
    """Return one image path per cluster, the one closest to its center.

    clustimage computes this itself in results_unique. Fall back to grouping
    by cluster label and keeping the first image of each if it is missing.
    """
    unique = getattr(cl, "results_unique", None)
    if unique is not None and unique.get("pathnames") is not None:
        return [
            Path(pathname)
            for label, pathname in zip(unique["labels"], unique["pathnames"])
            if pathname
        ]

    by_label = {}
    for label, pathname in zip(results["labels"], results["pathnames"]):
        by_label.setdefault(label, pathname)
    return [Path(pathname) for pathname in by_label.values()]


def unique_by_phash(paths, cutoff=0, keep=()):
    """Filter paths down to one image per perceptually-duplicate group.

    The clustering can under-split (two distinct stickers in one cluster
    survive only as its representative) or over-split (copies of one sticker
    in different clusters each get a representative). Seeding with the cluster
    representatives and then sweeping the whole corpus, keeping only images
    that are not perceptual duplicates of anything already kept, fixes both.
    """
    keep = list(dict.fromkeys(keep))
    kept = list(keep)
    kept_hashes = [imagehash.phash(Image.open(path)) for path in kept]

    keep_set = set(keep)
    for path in sorted(paths):
        if path in keep_set:
            continue
        phash = imagehash.phash(Image.open(path))
        if all(phash - kept_hash > cutoff for kept_hash in kept_hashes):
            kept.append(path)
            kept_hashes.append(phash)
    return kept


def copy_set(images, output_dir):
    """Copy the deduplicated set into output_dir."""
    written = []
    used_names = set()
    for pathname in images:
        dest = output_dir / pathname.name
        if dest.name in used_names:
            # Same basename from two different files: disambiguate instead
            # of overwriting.
            dest = output_dir / f"{dest.stem}-2{dest.suffix}"
            while dest.name in used_names:
                dest = output_dir / f"{dest.stem}2{dest.suffix}"
        shutil.copy2(pathname, dest)
        used_names.add(dest.name)
        written.append(dest)
    return written


def main():
    args = parse_args()
    if not args.images_dir.is_dir():
        sys.exit(f"Images directory not found: {args.images_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cl = Clustimage()

    imported_data = cl.import_data(str(args.images_dir))
    results = cl.fit_transform(
        imported_data, min_clust=args.min_clust, max_clust=args.max_clust
    )

    n_images = len(results["labels"])
    n_clusters = len(np.unique(results["labels"]))

    fig, _ = cl.clusteval.plot(
        savefig={"fname": str(args.output_dir / "clusteval.png"), "dpi": 120},
        showfig=False,
    )
    matplotlib.pyplot.close(fig)

    representatives = representative_pathnames(cl, results)
    all_paths = [Path(pathname) for pathname in results["pathnames"]]
    uniques = unique_by_phash(
        all_paths, cutoff=args.phash_cutoff, keep=representatives
    )
    written = copy_set(uniques, args.output_dir)

    print(
        f"\nClustered {n_images} images into {n_clusters} clusters; "
        f"wrote {len(written)} unique images to {args.output_dir}/"
    )


if __name__ == "__main__":
    main()