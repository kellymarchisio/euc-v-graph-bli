
def main():
    """
    1. Load fastText wiki data for relevant languages
    2. Load bilingual dictionaries for relvant language comparisons
    3. For each word pair (w1, w2),
        add directed edge e from w1 to w2 where w(e) = p(w2|w1) = p(w1, w2) / p(w1) (and vice versa)
    4. Run SGM on directed adjacency matrices
    5. Evaluate performance
    """
    pass


if __name__ == "__main__":
    main()
