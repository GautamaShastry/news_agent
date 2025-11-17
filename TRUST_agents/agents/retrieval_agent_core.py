# -*- coding: utf-8 -*-

"""
Improved Retrieval Agent Core
- BM25 hybrid with Dense retrieval (Sentence-Transformers + FAISS)
- Batched encoding, HNSW/Flat FAISS options, MMR re-ranking
- Robust PDF/HTML/txt ingestion with chunking (sentence-aware)
- Save/load index (FAISS index + embeddings + metadata)
- Configurable and documented API

FIXED: load_index() now properly initializes _dense_model for query encoding
"""

from __future__ import annotations
import os
import json
import math
import logging
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterable
from tqdm import tqdm

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# BM25
from rank_bm25 import BM25Okapi

# Dense retrieval
from sentence_transformers import SentenceTransformer
import numpy as np
try:
    import faiss
except Exception:
    faiss = None

# Parsing
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

# Optional re-ranker
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# Logging
logger = logging.getLogger("retrieval_agent_core")

# ---------------- Data classes ----------------
@dataclass
class Passage:
    id: str
    doc_id: str
    title: str
    text: str
    metadata: Dict[str, Any]

# ---------------- Utilities ----------------
def safe_read_text(path: str) -> str:
    encodings = ["utf-8", "latin-1", "utf-16"]
    for e in encodings:
        try:
            with open(path, "r", encoding=e, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    raise IOError(f"Cannot read {path} with known encodings")

def read_pdf_text(path: str) -> str:
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        try:
            ptext = page.extract_text()
        except Exception:
            ptext = None
        if ptext:
            text.append(ptext)
    return "\n".join(text)

def read_html_text(path_or_str: str, is_path: bool = True) -> str:
    if is_path:
        with open(path_or_str, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    else:
        raw = path_or_str
    soup = BeautifulSoup(raw, "html.parser")
    # Drop scripts/styles
    for s in soup(["script", "style", "header", "footer", "nav", "aside"]):
        s.extract()
    return soup.get_text(separator="\n")

def chunk_by_sentences(text: str, max_words: int = 160, overlap: int = 20) -> List[str]:
    sents = sent_tokenize(text)
    if not sents:
        return []
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        wcount = len(word_tokenize(s))
        if cur_len + wcount <= max_words or not cur:
            cur.append(s)
            cur_len += wcount
        else:
            chunks.append(" ".join(cur))
            # start new chunk with overlap sentences if needed
            if overlap > 0:
                # implement overlap by carrying last few words/sentences
                overlap_sents = []
                carry = 0
                # gather sentences from end until approx overlap words
                for sent in reversed(cur):
                    overlap_sents.insert(0, sent)
                    carry += len(word_tokenize(sent))
                    if carry >= overlap:
                        break
                cur = overlap_sents.copy()
                cur_len = sum(len(word_tokenize(x)) for x in cur)
            else:
                cur = []
                cur_len = 0
            cur.append(s)
            cur_len += wcount
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def batch_iter(iterable: Iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def cosine_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms

def mmr_rerank(query_emb: np.ndarray, candidate_embs: np.ndarray, candidate_scores: List[float],
               diversity: float = 0.7, top_k: int = 5) -> List[int]:
    """
    Simple MMR: chooses indices of candidates balancing score and diversity.
    Returns ordered list of selected indices (into candidate_embs).
    """
    selected = []
    candidate_embs = cosine_normalize(candidate_embs.copy())
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    sim_to_query = (candidate_embs @ query_emb.T).flatten().tolist()
    candidate_idxs = set(range(len(candidate_embs)))
    # initialize by top score
    scores = candidate_scores
    if not scores:
        return []
    first = int(np.argmax(scores))
    selected.append(first)
    candidate_idxs.remove(first)
    while len(selected) < min(top_k, len(candidate_embs)) and candidate_idxs:
        mmr_values = {}
        for idx in candidate_idxs:
            sim_q = sim_to_query[idx]
            sim_sel = max((candidate_embs[idx] @ candidate_embs[s].T).item() for s in selected) if selected else 0
            mmr_values[idx] = diversity * sim_q - (1 - diversity) * sim_sel
        next_idx = max(mmr_values.items(), key=lambda x: x[1])[0]
        selected.append(next_idx)
        candidate_idxs.remove(next_idx)
    return selected

# ---------------- RetrievalAgent ----------------
class RetrievalAgent:
    def __init__(
        self,
        index_dir: str = "retrieval_index",
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_name: Optional[str] = None,
        faiss_index_type: str = "Flat",  # options: Flat, HNSW, IVF
        faiss_dim: Optional[int] = None,
        device: str = "cpu",
    ):
        self.index_dir = pathlib.Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.dense_model_name = dense_model_name
        self.cross_encoder_name = cross_encoder_name
        self.faiss_index_type = faiss_index_type
        self.device = device

        self.passages: List[Passage] = []
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokenized: List[List[str]] = []
        self._dense_model: Optional[SentenceTransformer] = None
        self._faiss_index = None
        self._embeddings: Optional[np.ndarray] = None
        self._cross_encoder = None

    # ---------- ingestion ----------
    def index_corpus(self, corpus_path: str, chunk_max_words: int = 160, chunk_overlap: int = 20,
                     batch_size: int = 64, rebuild_dense: bool = True):
        """
        Walk directory (or single file) and create passages.
        Then build BM25 and optionally dense indices (if faiss + sentence-transformers available).
        """
        p = pathlib.Path(corpus_path)
        files = [p] if p.is_file() else list(p.rglob("*"))
        supported = {".txt", ".md", ".pdf", ".html", ".htm"}
        docs = [f for f in files if f.suffix.lower() in supported]
        logger.info("Found %d documents", len(docs))

        for f in tqdm(docs, desc="Parsing docs"):
            try:
                if f.suffix.lower() == ".pdf":
                    raw = read_pdf_text(str(f))
                elif f.suffix.lower() in {".html", ".htm"}:
                    raw = read_html_text(str(f), is_path=True)
                else:
                    raw = safe_read_text(str(f))
            except Exception as e:
                logger.warning("Skipping %s: %s", f, e)
                continue
            if not raw.strip():
                continue
            # chunk
            chunks = chunk_by_sentences(raw, max_words=chunk_max_words, overlap=chunk_overlap)
            for i, ch in enumerate(chunks):
                pid = f"{f.name}_{i}"
                meta = {"source_path": str(f), "chunk_index": i}
                self.passages.append(Passage(id=pid, doc_id=str(f), title=f.name, text=ch, metadata=meta))
        logger.info("Total passages: %d", len(self.passages))
        # build indices
        self.build_bm25()
        if rebuild_dense:
            self.build_dense(batch_size=batch_size)

    # ---------- BM25 ----------
    def build_bm25(self, tokenizer: Callable[[str], List[str]] = None):
        if tokenizer is None:
            tokenizer = lambda s: word_tokenize(s.lower())
        texts = [p.text for p in self.passages]
        tokenized = [tokenizer(t) for t in texts]
        self._bm25_tokenized = tokenized
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 built")

    # ---------- dense ----------
    def build_dense(self, batch_size: int = 64, use_gpu: bool = False):
        if faiss is None:
            logger.warning("FAISS not available. Skipping dense build.")
            return
        # load model
        self._dense_model = SentenceTransformer(self.dense_model_name, device=("cuda" if use_gpu else "cpu"))
        texts = [p.text for p in self.passages]
        logger.info("Encoding %d passages in batches (batch_size=%d)", len(texts), batch_size)
        embs_list = []
        for batch in tqdm(list(batch_iter(texts, batch_size)), desc="Encoding"):
            emb = self._dense_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embs_list.append(emb)
        embs = np.vstack(embs_list).astype("float32")
        embs = cosine_normalize(embs)
        self._embeddings = embs
        dim = embs.shape[1]
        logger.info("Embeddings shape: %s", embs.shape)

        # choose index type
        if self.faiss_index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 32)  # efConstruction default
            index.hnsw.efConstruction = 200
        elif self.faiss_index_type == "IVF":
            nlist = max(100, int(math.sqrt(len(embs))))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = min(10, nlist)
            index.train(embs)
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(embs)
        self._faiss_index = index
        logger.info("FAISS index built (%s)", self.faiss_index_type)

        # cross-encoder optional
        if self.cross_encoder_name and CrossEncoder is not None:
            self._cross_encoder = CrossEncoder(self.cross_encoder_name)
            logger.info("Cross-encoder loaded: %s", self.cross_encoder_name)
        elif self.cross_encoder_name:
            logger.warning("cross-encoder requested but package not available.")

    # ---------- save / load ----------
    def save_index(self, prefix: str = "index"):
        meta_path = self.index_dir / f"{prefix}_meta.json"
        emb_path = self.index_dir / f"{prefix}_emb.npy"
        faiss_path = self.index_dir / f"{prefix}_faiss.index"

        # metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in self.passages], f, ensure_ascii=False)
        # embeddings
        if self._embeddings is not None:
            np.save(str(emb_path), self._embeddings)
        # faiss
        if self._faiss_index is not None and faiss is not None:
            faiss.write_index(self._faiss_index, str(faiss_path))
        logger.info("Index saved in %s", self.index_dir)

    def load_index(self, prefix: str = "index", load_faiss: bool = True):
        """
        Load pre-built retrieval index from disk.
        
        FIXED: Now properly initializes _dense_model for query encoding.
        """
        meta_path = self.index_dir / f"{prefix}_meta.json"
        emb_path = self.index_dir / f"{prefix}_emb.npy"
        faiss_path = self.index_dir / f"{prefix}_faiss.index"

        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.passages = [Passage(**m) for m in meta]
        logger.info("Loaded %d passages metadata", len(self.passages))

        if load_faiss and faiss is not None and os.path.exists(faiss_path):
            self._faiss_index = faiss.read_index(str(faiss_path))
            if os.path.exists(emb_path):
                self._embeddings = np.load(str(emb_path))
            logger.info("Loaded FAISS index and embeddings (if present)")
            
            # ============================================================
            # CRITICAL FIX: Initialize dense model for query encoding
            # ============================================================
            # Without this, _dense_search() will return [] because 
            # _dense_model is None, causing 0 evidence to be retrieved.
            if self._dense_model is None:
                logger.info("Initializing dense model for query encoding: %s", self.dense_model_name)
                self._dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
                logger.info("✓ Dense model initialized successfully")
        
        # rebuild BM25
        self.build_bm25()

    # ---------- internal searches ----------
    def _bm25_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            return []
        tokens = word_tokenize(query.lower())
        scores = self._bm25.get_scores(tokens)
        idxs = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[int(i)])) for i in idxs if scores[int(i)] > 0]

    def _dense_search(self, query: str, k: int = 50) -> List[Tuple[int, float]]:
        if self._faiss_index is None or self._dense_model is None:
            return []
        q_emb = self._dense_model.encode([query], convert_to_numpy=True)
        q_emb = cosine_normalize(q_emb.astype("float32"))
        D, I = self._faiss_index.search(q_emb, k)
        idxs = I[0].tolist()
        scores = D[0].tolist()
        return [(int(i), float(s)) for i, s in zip(idxs, scores) if i >= 0]

    # ---------- public retrieve ----------
    def retrieve(self, query: str, top_k: int = 5, bm25_weight: float = 0.6,
                 candidate_k: int = 50, rerank_with_cross: bool = False,
                 mmr: bool = False, mmr_diversity: float = 0.7) -> List[Dict[str, Any]]:
        """
        Returns list of dicts:
        {passage_id, doc_id, title, text, metadata, bm25_score, dense_score, hybrid_score, cross_score}
        """
        bm25_res = self._bm25_search(query, k=candidate_k) if self._bm25 else []
        dense_res = self._dense_search(query, k=candidate_k) if self._faiss_index else []

        # merge scores in dict by index
        candidates = {}
        for idx, s in bm25_res:
            candidates[idx] = {"bm25": s, "dense": 0.0}
        for idx, s in dense_res:
            if idx in candidates:
                candidates[idx]["dense"] = s
            else:
                candidates[idx] = {"bm25": 0.0, "dense": s}

        if not candidates:
            return []

        # normalise scores
        all_scores = [v["bm25"] for v in candidates.values()] + [v["dense"] for v in candidates.values()]
        min_s, max_s = min(all_scores), max(all_scores)
        def norm(x):
            if max_s - min_s < 1e-9:
                return 0.0
            return (x - min_s) / (max_s - min_s)

        merged = []
        for idx, v in candidates.items():
            bm25_n = norm(v["bm25"])
            dense_n = norm(v["dense"])
            hybrid = bm25_weight * bm25_n + (1 - bm25_weight) * dense_n
            merged.append({"idx": idx, "bm25_score": v["bm25"], "dense_score": v["dense"], "hybrid_score": hybrid})

        merged = sorted(merged, key=lambda x: x["hybrid_score"], reverse=True)
        # optional MMR for diversity
        if mmr and self._embeddings is not None:
            candidate_idxs = [m["idx"] for m in merged[:candidate_k]]
            candidate_embs = self._embeddings[candidate_idxs]
            query_emb = self._dense_model.encode([query], convert_to_numpy=True)[0]
            selected_idxs = mmr_rerank(query_emb, candidate_embs, [m["hybrid_score"] for m in merged[:candidate_k]],
                                       diversity=mmr_diversity, top_k=top_k)
            merged = [merged[i] for i in selected_idxs]
        else:
            merged = merged[:top_k]

        # optional cross-encoder rerank
        if rerank_with_cross and self._cross_encoder is not None:
            pairs = [(query, self.passages[m["idx"]].text) for m in merged]
            scores = self._cross_encoder.predict(pairs)
            for m, s in zip(merged, scores):
                m["cross_score"] = float(s)
            merged = sorted(merged, key=lambda x: x.get("cross_score", x["hybrid_score"]), reverse=True)

        # produce output
        out = []
        for m in merged[:top_k]:
            p = self.passages[m["idx"]]
            out.append({
                "passage_id": p.id,
                "doc_id": p.doc_id,
                "title": p.title,
                "text": p.text,
                "metadata": p.metadata,
                "bm25_score": m.get("bm25_score"),
                "dense_score": m.get("dense_score"),
                "hybrid_score": m.get("hybrid_score"),
                "cross_score": m.get("cross_score", None)
            })
        return out