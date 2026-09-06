"""Lab 5 starter: two-stage bilingual case search."""


class CaseSearch:
    def __init__(self, prefix: str):
        # TODO(Lab 5): load manifest, index, metadata, encoder and reranker;
        # assert manifest integrity on load.
        raise NotImplementedError

    def search(self, query: str, k: int = 5, candidates: int = 50, min_score: float = 0.25):
        # TODO(Lab 5): normalise query, bi-encoder retrieval, CE rerank, honest empty result.
        raise NotImplementedError
