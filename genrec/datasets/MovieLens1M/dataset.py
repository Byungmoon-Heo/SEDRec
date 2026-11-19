# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import zipfile
import json
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

from genrec.dataset import AbstractDataset
from genrec.utils import download_file, clean_text


class MovieLens1M(AbstractDataset):
    """
    MovieLens 1M dataset loader for genrec.

    Contract (kept consistent with AmazonReviews2014):
      - self.all_item_seqs: dict[str, list[str]]  (raw user_id -> raw item_id sequence ordered by time)
      - self.id_mapping: dict with keys user2id, item2id, id2user, id2item (remapped IDs start at 1; 0 is [PAD])
      - self.item2meta: optional dict[str, Any] (raw item_id -> metadata), following config['metadata'] in {'none','raw','sentence'}
      - cached files under <cache_dir>/MovieLens1M/{raw,processed}/
    """

    ML1M_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def __init__(self, config: dict):
        super(MovieLens1M, self).__init__(config)

        self.log('[DATASET] MovieLens 1M')

        self.cache_dir = os.path.join(config['cache_dir'], 'MovieLens1M')
        self._download_and_process_raw()

    # --------------------------
    # Download / Extract
    # --------------------------
    def _download_raw(self, path: str) -> str:
        """
        Download and extract MovieLens 1M.

        Returns:
            str: local directory containing extracted files (ml-1m/)
        """
        zip_path = os.path.join(path, "ml-1m.zip")
        extract_dir = os.path.join(path, "ml-1m")

        # Download once
        if not os.path.exists(zip_path):
            self.log(f'[DATASET] Downloading: {self.ML1M_ZIP_URL}')
            download_file(self.ML1M_ZIP_URL, zip_path)
        else:
            self.log('[DATASET] Raw zip already exists...')

        # Extract once
        ratings_path = os.path.join(extract_dir, "ratings.dat")
        movies_path = os.path.join(extract_dir, "movies.dat")
        if os.path.exists(ratings_path) and os.path.exists(movies_path):
            self.log('[DATASET] Raw files already extracted...')
            return extract_dir

        self.log('[DATASET] Extracting ml-1m.zip ...')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # The zip contains a top-level "ml-1m/" folder
            zf.extractall(path)

        # Ensure expected paths
        if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
            raise RuntimeError(
                f"Extraction succeeded but expected files not found: {ratings_path}, {movies_path}. "
                "Please check the zip structure."
            )

        return extract_dir

    def _download_and_process_raw(self):
        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        with self.accelerator.main_process_first():
            raw_dir = self._download_raw(path=raw_data_path)

        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        ratings_path = os.path.join(raw_dir, "ratings.dat")
        movies_path = os.path.join(raw_dir, "movies.dat")

        self.all_item_seqs, self.id_mapping = self._process_ratings(
            input_path=ratings_path,
            output_path=processed_data_path,
        )

        self.item2meta = self._process_meta(
            input_path=movies_path,
            output_path=processed_data_path,
        )

    # --------------------------
    # Processing: interactions
    # --------------------------
    def _remap_ids(self, item_seqs: dict) -> Tuple[dict, dict]:
        """Remap raw user/item identifiers to contiguous integer ids.

        Same behavior/contract as AmazonReviews2014._remap_ids:
        - id 0 is reserved for '[PAD]'
        - new ids start from 1
        - self.all_item_seqs stores raw item ids (strings) in time order
        - self.id_mapping contains user2id/item2id/id2user/id2item
        """
        self.log('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)

            iids: List[str] = []
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)

            self.all_item_seqs[user] = iids

        return self.all_item_seqs, self.id_mapping

    def _load_ratings(self, path: str) -> List[Tuple[str, str, int]]:
        """
        Load ratings.dat.

        Format per line: UserID::MovieID::Rating::Timestamp
        Returns:
            list of (user_id, movie_id, timestamp)
        """
        out: List[Tuple[str, str, int]] = []
        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("::")
                if len(parts) != 4:
                    continue
                user, item, _rating, ts = parts
                try:
                    ts_i = int(ts)
                except Exception:
                    continue
                out.append((str(user), str(item), ts_i))
        return out

    def _k_core_filter(
        self,
        ratings: List[Tuple[str, str, int]],
        k: int,
    ) -> List[Tuple[str, str, int]]:
        """Iterative k-core filtering on the user-item interaction graph.

        Keeps only interactions where both:
          - user has >= k interactions
          - item has >= k interactions

        This mirrors the common "5-core" preprocessing used in RecSys papers.
        The procedure is deterministic (no randomness).
        """
        if k is None or k <= 1:
            return ratings

        cur = ratings
        while True:
            user_cnt = defaultdict(int)
            item_cnt = defaultdict(int)
            for u, it, _ts in cur:
                user_cnt[u] += 1
                item_cnt[it] += 1

            keep_users = {u for u, c in user_cnt.items() if c >= k}
            keep_items = {it for it, c in item_cnt.items() if c >= k}

            nxt = [(u, it, ts) for (u, it, ts) in cur if (u in keep_users and it in keep_items)]
            if len(nxt) == len(cur):
                break
            cur = nxt

        return cur

    def _get_item_seqs(self, ratings: List[Tuple[str, str, int]]) -> Dict[str, List[str]]:
        """
        Group by user and sort by timestamp.
        """
        item_seqs = defaultdict(list)
        for u, it, ts in ratings:
            item_seqs[u].append((ts, it))
        all_item_seqs: Dict[str, List[str]] = {}
        for u, pairs in item_seqs.items():
            pairs.sort(key=lambda x: x[0])
            all_item_seqs[u] = [it for _, it in pairs]
        return all_item_seqs

    def _process_ratings(self, input_path: str, output_path: str) -> Tuple[dict, dict]:
        """
        Process ratings.dat and cache:
          - all_item_seqs.json
          - id_mapping.json
        """
        seq_file = os.path.join(output_path, 'all_item_seqs.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            self.log('[DATASET] Ratings have been processed...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            return all_item_seqs, id_mapping

        self.log('[DATASET] Processing ratings...')

        ratings = self._load_ratings(input_path)

        # Apply k-core (default: 5-core) to match common sequential recommendation preprocessing.
        k_core = int(self.config.get('k_core', 5))
        ratings = self._k_core_filter(ratings, k=k_core)
        self.log(f'[DATASET] Applied {k_core}-core filtering. Remaining interactions: {len(ratings)}')
        item_seqs = self._get_item_seqs(ratings)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)

        self.log('[DATASET] Saving mapping data...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)

        return all_item_seqs, id_mapping

    # --------------------------
    # Processing: metadata
    # --------------------------
    def _load_movies(self, path: str, item2id: dict) -> dict:
        """
        Load movies.dat, filtered to items present in item2id.

        Format per line: MovieID::Title::Genres
        Returns:
            dict: raw_item_id -> {"title": ..., "genres": ...}
        """
        out = {}
        with open(path, "r", encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("::")
                if len(parts) != 3:
                    continue
                item, title, genres = parts
                item = str(item)
                if item not in item2id:
                    continue
                out[item] = {"title": title, "genres": genres}
        return out

    def _movies_to_sentence(self, meta: dict) -> dict:
        """
        Convert {"title","genres"} into a single clean sentence.
        """
        out = {}
        for item, m in meta.items():
            title = clean_text(m.get("title", "")) if isinstance(m, dict) else ""
            genres = ""
            if isinstance(m, dict):
                genres = m.get("genres", "")
            if isinstance(genres, str):
                genres = genres.replace("|", ", ")
            genres = clean_text(genres) if genres else ""

            # Sentence template
            sent = title
            if sent and not sent.endswith("."):
                sent += "."
            if genres:
                sent += f" Genres: {genres}."
            out[item] = sent.strip()
        return out

    def _process_meta(self, input_path: str, output_path: str) -> Optional[dict]:
        process_mode = self.config['metadata']
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
        if os.path.exists(meta_file):
            self.log('[DATASET] Metadata has been processed...')
            with open(meta_file, 'r') as f:
                return json.load(f)

        self.log(f'[DATASET] Processing metadata, mode: {process_mode}...')

        if process_mode == 'none':
            return None

        item2meta = self._load_movies(path=input_path, item2id=self.item2id)

        if process_mode == 'raw':
            pass
        elif process_mode == 'sentence':
            item2meta = self._movies_to_sentence(item2meta)
        else:
            raise NotImplementedError('Metadata processing type not implemented.')

        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)

        return item2meta
