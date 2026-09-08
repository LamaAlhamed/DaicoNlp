"""Lab 3 starter: NER label alignment."""


def align_labels(word_ids, word_labels):
    """Align word-level BIO label ids to subword token positions.

    - Special tokens (``word_id is None``, e.g. [CLS]/[SEP]/padding) -> -100.
    - The FIRST subword piece of a word gets that word's label.
    - Any CONTINUATION subword piece of the same word (including Arabic
      clitic segmentation, which can split one orthographic word into
      several subword pieces) is masked with -100, so the loss is only
      computed once per word.
    """
    aligned = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word_id:
            aligned.append(word_labels[word_id])
        else:
            aligned.append(-100)
        previous_word_id = word_id
    return aligned
