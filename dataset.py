import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pyarrow as pa
import pyarrow.dataset as ds
import torch
from pyfaidx import Fasta
from torch.utils.data import IterableDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

TEST_CHROMS  = ["chr17"]
VAL_CHROMS   = ["chr20", "chr21", "chr22"]
ROLE2ID = {"TSS":0, "D":1, "A":2, "PAS":3}
RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    return seq.translate(RC)[::-1]


def split_filters_by_chroms():
    test = ds.field("chrom").isin(TEST_CHROMS)
    val = ds.field("chrom").isin(VAL_CHROMS)
    train = ~(ds.field("chrom").isin(TEST_CHROMS + VAL_CHROMS))
    return train, val, test


def _splice_sites(strand, xs, xe):
    """
    Return [pos0, pos1, ...] in TRANSCRIPT order and their roles.
    roles sequence is [TSS, D, A, D, ..., PAS].
    """
    n = len(xs)
    roles, pos = [], []

    for i in range(n):
        if strand == "+":
            a_pos, d_pos = xs[i], xe[i]   # genomic left -> right
        else:
            # For '-' strand, transcript runs from higher to lower genomic coords.
            # In transcript order, exon boundary positions are [end, start].
            a_pos, d_pos = xe[i], xs[i]   # genomic high -> low

        roles.append(ROLE2ID["A"]); pos.append(int(a_pos))
        roles.append(ROLE2ID["D"]); pos.append(int(d_pos))
        
    roles[0] = ROLE2ID["TSS"]
    roles[-1] = ROLE2ID["PAS"]
    return roles, pos


class GeneIterableFixedB(IterableDataset):
    def __init__(
        self,
        parquet_dir: str | Path,
        split_filter,
        tx_batch_size: int,
        pad_bp: int = 5000,
        shuffle: bool = True,
        max_locus_len: Optional[int] = 65536,
        arrow_batch_size: int = 128,
    ):
        super().__init__()
        self.tx_bs = int(tx_batch_size)
        self.pad = int(pad_bp)
        self.shuffle = shuffle
        self.max_len = max_locus_len
        self.arrow_batch_size = int(arrow_batch_size)
        self.split = split_filter

        # LRU-style cache for locus embeddings
        self._emb_cache = OrderedDict()

        # List all parquet files; we will shuffle this per epoch
        parquet_dir = Path(parquet_dir)
        self.parquet_paths = sorted(parquet_dir.glob("*.parquet"))

        # Caduceus model
        self.model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.GRCh38 = Fasta("data/genome_assembly/GRCh38.primary_assembly.genome.fa")

        # Caduceus embedding dim (must match PointerHead d_emb)
        self.d_emb = 256

    def _iter_genes_from_file(self, path: Path, rng: random.Random) -> Iterator[Dict[str, Any]]:
        """
        Stream genes from a single parquet file, shuffling rows within each
        Arrow record batch.
        """
        cols = ["gene_id", "chrom", "start", "end", "strand", "transcripts"]
        dset = ds.dataset(str(path), format="parquet")

        scanner = dset.scanner(
            filter=self.split,
            columns=cols,
            use_threads=True,
            batch_size=self.arrow_batch_size,
        )
        reader = scanner.to_reader()
        for rb in reader:
            tbl = pa.Table.from_batches([rb])
            rows = tbl.to_pylist()
            if self.shuffle:
                rng.shuffle(rows)
            for r in rows:
                yield r

    @staticmethod
    def _locus_window(s: int, e: int, pad: int) -> tuple[int, int]:
        return max(0, int(s) - pad), int(e) + pad

    def _tx_examples(self, g, s, e):
        """
        Build per-transcript training examples.

        Each transcript yields:
          - One TSS example with empty history (if TSS is inside locus)
          - A chain of boundary examples:
              prev (TSS / D / A / ...)  -> next (D / A / PAS)
        Includes per-example rel_abundance inherited from the transcript.
        """
        out = []
        strand = g["strand"]
        tx_list = g.get("transcripts")

        for tx in tx_list:
            xs = [int(x) - 1 for x in tx["exon_starts"]]
            xe = [int(x) - 1 for x in tx["exon_ends"]]
            if not xs or not xe or len(xs) != len(xe):
                continue

            roles, pos = _splice_sites(strand, xs, xe)
            rel_ab = float(tx.get("rel_abundance", 1.0))

            # ---------- TSS example with empty history ----------
            tss_abs = int(pos[0])
            tss_role = int(roles[0])  # ROLE2ID["TSS"]

            if s <= tss_abs < e:
                if strand == "+":
                    tss_idx = tss_abs - s
                else:
                    tss_idx = (e - 1) - tss_abs

                L = e - s
                if 0 <= tss_idx < L:
                    out.append({
                        "prev_idx": [],
                        "prev_roles": [],
                        "next_idx": int(tss_idx),
                        "next_role": tss_role,
                        "rel_abundance": rel_ab,
                    })

            # ---------- Normal chain examples (TSS -> D -> A -> ... -> PAS) ----------
            prev_idx, prev_roles = [], []

            for i in range(len(pos) - 1):
                ctx_abs = int(pos[i])
                nxt_abs = int(pos[i + 1])
                curr_role = roles[i]
                nxt_role  = roles[i + 1]

                if not (s <= ctx_abs < e and s <= nxt_abs < e):
                    continue

                if strand == "+":
                    ctx_idx = ctx_abs - s
                    nxt_idx = nxt_abs - s
                else:
                    ctx_idx = (e - 1) - ctx_abs
                    nxt_idx = (e - 1) - nxt_abs

                if nxt_idx <= ctx_idx:
                    continue

                prev_idx.append(int(ctx_idx))
                prev_roles.append(int(curr_role))

                out.append({
                    "prev_idx": prev_idx.copy(),
                    "prev_roles": prev_roles.copy(),
                    "next_idx": int(nxt_idx),
                    "next_role": int(nxt_role),
                    "rel_abundance": rel_ab,
                })

        return out

    def __iter__(self):
        """
        per-epoch file-level shuffle + in-batch row shuffle.
        """
        rng = random.Random()
        file_list = self.parquet_paths[:]

        if self.shuffle:
            rng.shuffle(file_list)

        carry = []
        carry_gene_keys = []

        for path in file_list:
            for g in self._iter_genes_from_file(path, rng):
                chrom, strand = g["chrom"], g["strand"]
                gs, ge = int(g["start"]) - 1, int(g["end"])
                s, e = self._locus_window(gs, ge, self.pad)
                if self.max_len is not None and (e - s) > self.max_len:
                    continue

                txs = self._tx_examples(g, s, e)
                if not txs:
                    continue

                # Cache key: (chrom, strand, s, e, gene_id)
                gene_key = (chrom, strand, s, e, g["gene_id"])
                for t in txs:
                    carry.append(t)
                    carry_gene_keys.append(gene_key)

                while len(carry) >= self.tx_bs:
                    batch_items = carry[: self.tx_bs]
                    batch_gkeys = carry_gene_keys[: self.tx_bs]
                    carry = carry[self.tx_bs :]
                    carry_gene_keys = carry_gene_keys[self.tx_bs :]
                    yield self._pack_batch(batch_items, batch_gkeys)

        if carry:
            yield self._pack_batch(carry, carry_gene_keys)

    @torch.no_grad()
    def _embed_locus(self, key):
        """
        Embed a locus with Caduceus and store as float16 on CPU.
        """
        chrom, strand, s, e, gene_id = key
        if key in self._emb_cache:
            emb = self._emb_cache.pop(key)
            self._emb_cache[key] = emb
            return emb

        L = int(e) - int(s)
        if L <= 0:
            # zero-length locus: return empty fp16 tensor
            return torch.zeros(0, self.d_emb, dtype=torch.float16)

        seq = str(self.GRCh38[chrom][s:e])
        if len(seq) != (e - s):
            raise RuntimeError(
                f"FASTA slice truncated: chrom={chrom} s={s} e={e} "
                f"requested={e-s} got={len(seq)}"
            )
        if strand == "-":
            seq = reverse_complement(seq)

        tokens = self.tokenizer(seq, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model(**tokens, output_hidden_states=True)
        # store as fp16 on CPU to save memory
        h = out.hidden_states[-1].squeeze(0).detach().to("cpu", dtype=torch.float16)  # [L, D_emb]

        self._emb_cache[key] = h
        if len(self._emb_cache) > self.tx_bs:
            self._emb_cache.popitem(last=False)
        return h

    def _pack_batch(self, items, gkeys):
        B = len(items)
        ctx_lengths, suffix_lengths = [], []
        tgt_idx = torch.empty(B, dtype=torch.long)
        rel_ab = torch.empty(B, dtype=torch.float32)
        
        # 1. Pre-scan for dimensions
        for i, (it, key) in enumerate(zip(items, gkeys)):
            chrom, strand, s, e, gene_id = key
            L = int(e) - int(s)

            prev_idx = it["prev_idx"]
            ctx_len = len(prev_idx)
            ctx_lengths.append(ctx_len)

            if ctx_len > 0:
                c = int(prev_idx[-1])
            else:
                c = -1

            n = int(it["next_idx"])

            tgt_idx[i] = n - (c + 1)
            suffix_lengths.append(L - (c + 1))
            rel_ab[i] = float(it.get("rel_abundance", 1.0))

        max_ctx = max(ctx_lengths) if ctx_lengths else 0
        S = max(suffix_lengths)
        D = self.d_emb

        # 2. Allocate Tensors (embeddings in fp16)
        X_context     = torch.zeros(B, max_ctx, D, dtype=torch.float16)
        context_pos   = torch.zeros(B, max_ctx, dtype=torch.long)
        context_roles = torch.zeros(B, max_ctx, dtype=torch.long)
        context_mask  = torch.zeros(B, max_ctx, dtype=torch.float32)
        
        X_suffix    = torch.zeros(B, S, D, dtype=torch.float16)
        suffix_pos  = torch.zeros(B, S, dtype=torch.long)
        suffix_mask = torch.zeros(B, S, dtype=torch.float32)
        
        target_role = torch.zeros(B, dtype=torch.long)

        # 3. Fill
        for i, (it, key, ctx_len, suf_len) in enumerate(
            zip(items, gkeys, ctx_lengths, suffix_lengths)
        ):
            emb = self._embed_locus(key)  # (L, D_emb) in fp16
            prev_idx = it["prev_idx"]

            if ctx_len > 0:
                c = int(prev_idx[-1])
            else:
                c = -1

            if ctx_len > 0:
                ctx_indices_t = torch.tensor(prev_idx, dtype=torch.long)
                X_context[i, :ctx_len] = emb[ctx_indices_t]
                context_pos[i, :ctx_len] = ctx_indices_t
                context_roles[i, :ctx_len] = torch.tensor(it["prev_roles"], dtype=torch.long)
                context_mask[i, :ctx_len] = 1.0

            if suf_len > 0:
                seg = emb[c + 1 : c + 1 + suf_len]
                X_suffix[i, :suf_len] = seg
                suffix_pos[i, :suf_len] = torch.arange(
                    c + 1, c + 1 + suf_len, dtype=torch.long
                )
                suffix_mask[i, :suf_len] = 1.0

            target_role[i] = it["next_role"]

        return {
            "X_context":      X_context,      # (B, M, D_emb) fp16
            "context_pos":    context_pos,    # (B, M)
            "context_roles":  context_roles,  # (B, M)
            "context_mask":   context_mask,   # (B, M) fp32
            "X_suffix":       X_suffix,       # (B, S, D_emb) fp16
            "suffix_pos":     suffix_pos,     # (B, S)
            "suffix_mask":    suffix_mask,    # (B, S) fp32
            "target_idx":     tgt_idx,        # (B,)
            "target_role":    target_role,    # (B,)
            "rel_abundance":  rel_ab,         # (B,) fp32
        }
