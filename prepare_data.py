import argparse
import contextlib
import json
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


class _DummyAccelerator:
    is_main_process: bool = True

    @contextlib.contextmanager
    def main_process_first(self):
        yield


def _get_accelerator():
    try:
        from accelerate import Accelerator 
        return Accelerator()
    except Exception:
        return _DummyAccelerator()


def _sanitize_tag(model_name: str) -> str:
    tag = model_name.strip()
    tag = tag.replace("sentence-transformers/", "")
    tag = tag.replace("/", "-")
    tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", tag)
    return tag


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_or_build_genrec_dataset(
    root: str,
    dataset: str,
    category: Optional[str],
    cache_dir: str,
    metadata: str,
    k_core: int = 5,
):
    sys.path.insert(0, root)

    accel = _get_accelerator()
    base_cfg = {
        "cache_dir": cache_dir,
        "metadata": metadata,          # none / raw / sentence
        "split": "leave_one_out",
        "accelerator": accel,
    }

    if dataset == "AmazonReviews2014":
        if not category:
            raise ValueError("--category is required when --dataset AmazonReviews2014")
        try:
            from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014 
        except Exception as e:
            raise RuntimeError(
                "Failed to import genrec from the provided --root. "
            ) from e
        cfg = {**base_cfg, "category": category}
        return AmazonReviews2014(cfg)

    if dataset == "MovieLens1M":
        try:
            from genrec.datasets.MovieLens1M.dataset import MovieLens1M  
        except Exception as e:
            raise RuntimeError(
                "Failed to import MovieLens1M dataset from genrec. "
            ) from e
        cfg = dict(base_cfg)
        cfg["k_core"] = int(k_core)
        return MovieLens1M(cfg)

    raise ValueError(f"Unknown dataset: {dataset}")

def _build_sequence_splits(all_item_seqs_raw: Dict[str, List[str]], item2id: Dict[str, int], user2id: Optional[Dict[str, int]] = None, min_len: int = 3,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:

    kept: List[Tuple[str, List[int]]] = []
    for raw_user, raw_items in all_item_seqs_raw.items():
        seq_ids = [item2id[it] for it in raw_items if it in item2id]
        if len(seq_ids) >= min_len:
            kept.append((raw_user, seq_ids))

    if user2id is not None:
        kept.sort(key=lambda x: user2id.get(x[0], 10**18))
    else:
        kept.sort(key=lambda x: x[0])

    raw_user_to_new: Dict[str, int] = {u: idx for idx, (u, _) in enumerate(kept)}

    train: Dict[int, List[int]] = {}
    val: Dict[int, List[int]] = {}
    test: Dict[int, List[int]] = {}

    for raw_user, seq in kept:
        uid = raw_user_to_new[raw_user]
        train[uid] = seq[:-2]
        val[uid] = [seq[-2]]
        test[uid] = [seq[-1]]

    return train, val, test


def _collect_item_sentences(id2item: List[str], item2meta: Optional[Dict[str, str]],) -> Tuple[List[str], Dict[str, str], int]:
    n_items = len(id2item) - 1  
    sentences: List[str] = []
    mapping: Dict[str, str] = {}
    missing = 0

    for iid in range(1, n_items + 1):
        raw_item = id2item[iid]
        sent = ""
        if item2meta is not None:
            sent = item2meta.get(raw_item, "")
        if not isinstance(sent, str):
            sent = str(sent)
        sent = sent.strip()
        if sent == "":
            missing += 1
        sentences.append(sent)
        mapping[str(iid)] = sent

    return sentences, mapping, missing


def _embed_sentences_sentence_transformers(
    sentences: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is not available. Install it or use --emb_backend openai."
        ) from e

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    emb = np.asarray(emb, dtype=np.float32)
    return emb


def _embed_sentences_openai(
    sentences: List[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """
    Requires OPENAI_API_KEY in environment and the official openai python package.
    """
    try:
        from openai import OpenAI 
    except Exception as e:
        raise RuntimeError(
            "openai python package is not available. Install it or use --emb_backend sentence-transformers."
        ) from e

    client = OpenAI()
    out: List[List[float]] = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="OpenAI embeddings"):
        batch = sentences[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        
        for d in resp.data:
            out.append(d.embedding)

    emb = np.asarray(out, dtype=np.float32)
    return emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="AmazonReviews2014", choices=["AmazonReviews2014", "MovieLens1M"])
    p.add_argument("--category", type=str, default=None, help="AmazonReviews2014 category (e.g., Beauty). Ignored for MovieLens1M.")
    p.add_argument("--diffurec_out_dir", type=str, required=True, help="Output dataset directory.")
    p.add_argument("--cache_dir", type=str, default=None, help="Where genrec will cache raw/processed.")
    p.add_argument("--metadata", type=str, default="sentence", choices=["none", "raw", "sentence"], help="genrec metadata processing mode. For semantic embedding, use 'sentence'.")
    p.add_argument("--min_seq_len", type=int, default=3, help="Minimum full sequence length to keep a user")
    p.add_argument("--k_core", type=int, default=5, help="k-core filtering threshold. Used only for MovieLens1M; ignored for AmazonReviews2014.")
    p.add_argument("--emb_backend", type=str, default="sentence-transformers")
    p.add_argument("--emb_model", type=str, default="sentence-transformers/sentence-t5-base", help="Embedding model name.")
    p.add_argument("--emb_batch_size", type=int, default=256, help="Embedding batch size.")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for sentence-transformers backend.")
    p.add_argument("--semantic_out_name", type=str, default=None, help="Output .npy file name.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")

    args = p.parse_args()

    diffu_out = os.path.abspath(args.diffurec_out_dir)
    processed_out = os.path.join(diffu_out, "processed")
    _ensure_dir(diffu_out)
    _ensure_dir(processed_out)

    cache_dir = args.cache_dir or os.path.join(diffu_out, "_genrec_cache")
    _ensure_dir(cache_dir)

    ds = _load_or_build_genrec_dataset(
        root=os.path.abspath(args.root),
        dataset=args.dataset,
        category=args.category,
        cache_dir=cache_dir,
        metadata=args.metadata,
        k_core=args.k_core,
    )

    train, val, test = _build_sequence_splits(
        all_item_seqs_raw=ds.all_item_seqs,
        item2id=ds.item2id,
        user2id=getattr(ds, "user2id", None),
        min_len=args.min_seq_len,
    )

    id2item: List[str] = ds.id_mapping["id2item"]
    n_items = len(id2item) - 1
    n_users = len(train)

    smap = {i: i for i in range(1, n_items + 1)}
    umap = {i: i - 1 for i in range(1, n_users + 1)}

    dataset_pkl_path = os.path.join(diffu_out, "dataset.pkl")
    if os.path.exists(dataset_pkl_path) and not args.overwrite:
        print(f"[SKIP] dataset.pkl already exists: {dataset_pkl_path} (use --overwrite to replace)")
    else:
        payload = {"train": train, "val": val, "test": test, "umap": umap, "smap": smap}
        with open(dataset_pkl_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[OK] Wrote: {dataset_pkl_path}")

    item2meta = getattr(ds, "item2meta", None)
    if args.metadata != "sentence":
        print("[WARN] metadata != 'sentence'. item2meta may be None or non-sentence; embeddings may be poor/empty.")

    sentences, meta_json, missing = _collect_item_sentences(id2item=id2item, item2meta=item2meta)
    print(f"[INFO] Items: {n_items}, Users kept: {n_users}, Missing/empty meta sentences: {missing}")

    meta_json_path = os.path.join(processed_out, "metadata.sentence.json")
    if (not os.path.exists(meta_json_path)) or args.overwrite:
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_json, f, ensure_ascii=False)
        print(f"[OK] Wrote: {meta_json_path}")

    tag = _sanitize_tag(args.emb_model)
    out_name = args.semantic_out_name or f"semantic_emb_{tag}.npy"
    sem_path = os.path.join(processed_out, out_name)

    if os.path.exists(sem_path) and not args.overwrite:
        print(f"[SKIP] semantic embedding already exists: {sem_path} (use --overwrite to replace)")
    else:
        if args.emb_backend == "sentence-transformers":
        # if args.emb_backend == "all-mpnet-base-v2":
            emb_1_to_n = _embed_sentences_sentence_transformers(
                sentences=sentences,
                model_name=args.emb_model,
                batch_size=args.emb_batch_size,
                device=args.device,
            )
        
        else:
            emb_1_to_n = _embed_sentences_openai(
                sentences=sentences,
                model_name=args.emb_model,
                batch_size=args.emb_batch_size,
            )

        if emb_1_to_n.shape[0] != n_items:
            raise RuntimeError(f"Embedding rows mismatch: got {emb_1_to_n.shape[0]} vs n_items={n_items}")

        sem = np.zeros((n_items + 1, emb_1_to_n.shape[1]), dtype=np.float32)
        sem[1:, :] = emb_1_to_n
        np.save(sem_path, sem)
        print(f"[OK] Wrote: {sem_path}  shape={sem.shape} dtype={sem.dtype}")

    if args.dataset == "AmazonReviews2014":
        genrec_proc = os.path.join(cache_dir, "AmazonReviews2014", args.category, "processed")
    else:
        genrec_proc = os.path.join(cache_dir, "MovieLens1M", "processed")

    for fname in ["id_mapping.json", "all_item_seqs.json"]:
        src = os.path.join(genrec_proc, fname)
        if os.path.exists(src):
            dst = os.path.join(processed_out, fname)
            if (not os.path.exists(dst)) or args.overwrite:
                shutil.copy2(src, dst)
                print(f"[OK] Copied: {dst}")
        else:
            print(f"[WARN] Not found (skip copy): {src}")

    print("[DONE]")


if __name__ == "__main__":
    main()
