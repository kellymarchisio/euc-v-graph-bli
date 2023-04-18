import numpy as np
import json
import math
from pathlib import Path
import tqdm
import itertools
import tensorflow_datasets as tfds


def process_dict_pairs(pair_file):
    """Parses a dictionary pairs file.

    Pairs as list of (srcwd, trgwd) tuples
    L1 and L2 vocabularies as sets.
    """
    pairs = []
    l1_words = set()
    l2_words = set()
    with open(pair_file) as f:
        for line in f:
            w1, w2 = line.split()
            w1 = w1.strip()
            w2 = w2.strip()
            pairs.append((w1, w2))
            l1_words.add(w1)
            l2_words.add(w2)
    return pairs, l1_words, l2_words


def load_wiki_data(lang):
    """
    Load fastText wiki data for relevant languages
    """
    # TODO remove [:1%]
    # load wiki40b data for language
    ds = tfds.load(f"wiki40b/{lang}", split="train[:1%]")

    # separate samples by special markers
    special_markers = [
        "_START_ARTICLE_",
        "_START_SECTION_",
        "_START_PARAGRAPH_",
        "_NEWLINE_",
    ]

    # TODO add tqdm
    wiki_data = []
    for example in ds:
        text = example["text"].numpy().decode("utf-8")
        text = text.replace("\n", "")
        for marker in special_markers:
            text = text.replace(marker, "[SEP]")
        docs = list(filter(lambda x: x != "", text.split("[SEP]")))
        wiki_data.extend(docs)

    return wiki_data


def main():
    """
    1. Load bilingual dictionaries for relvant language comparisons
    2. Load fastText wiki data for each language
    3. For each word pair (w1, w2) in each language:
        add directed edge e from w1 to w2 where w(e) = p(w2|w1) = p(w1, w2) / p(w1) (and vice versa)
    4. Run SGM on directed adjacency matrices
    5. Evaluate performance
    """
    src = "en"
    trg = "de"

    word_pairs, src_words, trg_words = process_dict_pairs(
        f"dicts/{src}-{trg}/train/{src}-{trg}.0-5000.txt.1to1"
    )

    # TODO remove
    src_words = list(src_words)[:10]
    trg_words = list(trg_words)[:10]

    for lang, words in [(src, src_words), (trg, trg_words)]:
        wiki_data = load_wiki_data(lang)

        # TODO remove
        wiki_data = wiki_data[:10000]

        unigram_path = Path(f"unigram_counts/{lang}.json")
        unigram_path.parent.mkdir(parents=True, exist_ok=True)

        if unigram_path.exists():
            print(f"Loading unigram counts for {lang}")
            with open(unigram_path) as f:
                unigram_counts = json.load(f)
        else:
            print(f"Computing unigram counts for {lang}")

            unigram_counts = {}
            for w in tqdm.tqdm(words):
                # compute monogram count for "w"
                count = 0
                for doc in wiki_data:
                    count += doc.count(w)
                unigram_counts[w] = count

            # save unigram counts
            with open(unigram_path, "w") as f:
                json.dump(unigram_counts, f, indent=4)

        bigram_path = Path(f"bigram_counts/{lang}.json")
        bigram_path.parent.mkdir(parents=True, exist_ok=True)

        if bigram_path.exists():
            print(f"Loading bigram counts for {lang}")
            with open(bigram_path) as f:
                bigram_counts = json.load(f)
        else:
            print(f"Computing bigram counts for {lang}")

            bigram_counts = {}
            for w1, w2 in tqdm.tqdm(
                itertools.permutations(words, 2), total=math.perm(len(words), 2)
            ):
                # compute bigram count for "w1 w2"
                count = 0
                for doc in wiki_data:
                    count += doc.count(f"{w1} {w2}")
                bigram_counts[(w1, w2)] = count

            # save bigram counts
            with open(bigram_path, "w") as f:
                json.dump(bigram_counts, f, indent=4)

        # create directed adjacency matrix
        adj_matrix_path = Path(f"adj_matrices/{lang}.json")
        adj_matrix_path.parent.mkdir(parents=True, exist_ok=True)

        if adj_matrix_path.exists():
            print(f"Loading adjacency matrix for {lang}")
            with open(adj_matrix_path) as f:
                adj_matrix = json.load(f)
        else:
            print(f"Computing adjacency matrix for {lang}")

            adj_matrix = np.zeros((len(words), len(words)))
            for w1, w2 in tqdm.tqdm(
                itertools.permutations(words, 2), total=math.perm(len(words), 2)
            ):
                pass

            # save bigram counts
            with open(bigram_path, "w") as f:
                json.dump(adj_matrix, f, indent=4)


if __name__ == "__main__":
    main()
