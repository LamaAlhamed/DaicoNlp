"""Lab 3 starter: extractive QA post-processing."""
import numpy as np


def best_span(start_logits, end_logits, offsets, *, null_score, null_threshold, max_answer_len=30, top_k=20):
    """Pick the best valid (start, end) span, or return an honest null answer.

    - Only positions with a real ``offsets[i]`` (context tokens, not
      [CLS]/[SEP]/question tokens) are considered.
    - A span is only valid if ``end_index >= start_index`` (never invert the
      order, even if that combination happens to score higher) and the
      character length is within ``max_answer_len``.
    - We only look at the top-``top_k`` start/end positions by logit score,
      like standard SQuAD-style post-processing, then take the highest
      scoring *valid* combination among them.
    - If the best valid span's score does not beat ``null_score`` by at
      least ``null_threshold``, we honestly return no answer instead of
      forcing a low-confidence guess.
    """
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    valid_indices = [i for i in range(len(offsets)) if offsets[i] is not None]

    def top(indices, logits):
        return sorted(indices, key=lambda i: logits[i], reverse=True)[:top_k]

    start_candidates = top(valid_indices, start_logits)
    end_candidates = top(valid_indices, end_logits)

    best = None
    for start_index in start_candidates:
        for end_index in end_candidates:
            if end_index < start_index:
                continue  # reject inverted spans
            start_char, _ = offsets[start_index]
            _, end_char = offsets[end_index]
            if end_char - start_char > max_answer_len:
                continue
            score = float(start_logits[start_index] + end_logits[end_index])
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_char": start_char,
                    "end_char": end_char,
                }

    null_score = float(null_score)

    if best is None or (null_score - best["score"]) >= null_threshold:
        return {"answer": None, "score": null_score, "null_score": null_score}

    return {
        "answer": (best["start_char"], best["end_char"]),
        "start_index": best["start_index"],
        "end_index": best["end_index"],
        "score": best["score"],
        "null_score": null_score,
    }
