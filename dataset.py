# gene_iter_fixedB.py
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator, Callable
import random
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
from pyfaidx import Fasta


TEST_CHROMS  = ["chr1", "chr2"]
VAL_CHROMS   = ["chr3", "chr4"]

RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def reverse_complement(seq: str) -> str:
    return seq.translate(RC)[::-1]

def split_filters_by_chroms():
    test = ds.field("chrom").isin(TEST_CHROMS)
    val = ds.field("chrom").isin(VAL_CHROMS)
    train = ~(ds.field("chrom").isin(TEST_CHROMS + VAL_CHROMS))
    return train, val, test

# ---------- Pointer roles ----------
ROLE2ID = {"TSS":0, "D":1, "A":2, "PAS":3}
PAD_ROLE = -1

def _splice_sites(strand, xs, xe):
    """
    Return [(A,pos), (D,pos), (A,pos), (D,pos), ...] in TRANSCRIPT order.
    xs/xe are exon starts/ends in genomic ascending order.
    """
    n = len(xs)

    if strand == "+":
        order = range(n)                 # exon0, exon1, ... (transcript = genomic)
        exon_start = lambda i: xs[i]     # A = exon start
        exon_end   = lambda i: xe[i]     # D = exon end
    else:
        order = range(n - 1, -1, -1)     # exon_{n-1}, ..., exon0 (transcript reverse of genomic)
        exon_start = lambda i: xe[i]     # on '-' strand, transcript start of exon is genomic END
        exon_end   = lambda i: xs[i]     # on '-' strand, transcript end   of exon is genomic START

    roles, pos = [], []
    for i in order:
        roles.append(ROLE2ID["A"]); pos.append(int(exon_start(i)))  # start of exon
        roles.append(ROLE2ID["D"]); pos.append(int(exon_end(i)))    # end   of exon
    return roles, pos

class GeneIterableFixedB(IterableDataset):
    """
    Stream Parquet genes; create constant-transcripts-per-batch (tx_batch_size).
    If a gene has >B transcripts, it spans multiple consecutive batches;
    if it underfills, subsequent genes top off the batch.

    Each yielded batch:
        roles      : LongTensor [B, S] with PAD_ROLE = -1 where padded
        positions  : LongTensor [B, S] locus-relative (>=0); masked where PAD
        mask       : FloatTensor [B, S] (1.0 = valid, 0.0 = pad)
        groups     : List[dict] per unique gene in the batch:
                     { 'gene_key', 'indices', 'locus_embedding', 'locus_start', 'locus_end' }
    """

    def __init__(
        self,
        parquet_dir: str | Path,
        split_filter,  # pyarrow expression, e.g., ds.field("split") == "train"
        tx_batch_size: int,
        pad_bp: int = 5000,
        shuffle: bool = True,
        max_locus_len: Optional[int] = None,
        arrow_batch_size: int = 128,       # Arrow record-batch size
    ):
        super().__init__()
        self.tx_bs = int(tx_batch_size)
        self.pad = int(pad_bp)
        self.shuffle = shuffle
        self.max_len = max_locus_len
        self.arrow_batch_size = int(arrow_batch_size)

        self.dset = ds.dataset(str(parquet_dir), format="parquet")
        self.split = split_filter

        self._emb_cache = OrderedDict()       # gene_key -> torch.Tensor [L, D]

        self.model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.human = Fasta("data/genome_assembly/GRCh38.primary_assembly.genome.fa")

    # --------- streaming gene iterator ---------

    def _iter_genes(self) -> Iterator[Dict[str, Any]]:
        """
        Stream genes from Parquet with bounded memory.
        """
        cols = ["gene_id", "chrom", "start", "end", "strand", "transcripts"]
        scanner = self.dset.scanner(
            filter=self.split,
            columns=cols,
            use_threads=True,
            batch_size=self.arrow_batch_size,
        )
        reader = scanner.to_reader()

        for rb in reader:
            tbl = pa.Table.from_batches([rb])
            rows = tbl.to_pylist()  # only this record-batch into Python
            if self.shuffle:
                random.shuffle(rows)   # per-batch shuffle (cheap)
            
            for r in rows:
                yield r

    @staticmethod
    def _locus_window(s: int, e: int, pad: int) -> tuple[int, int]:
        """Add symmetric padding to the gene window (clamped at 0)."""
        return max(0, int(s) - pad), int(e) + pad

    def _tx_examples(self, g: Dict[str, Any], locus_start: int, locus_end: int) -> List[Dict[str, Any]]:
        """
        Emit one example per context site (D/A) that has a *next* site inside the locus.
        Each example carries:
        - context_pos: absolute coord of context site
        - next_pos   : absolute coord of next site (D/A/PAS)
        We filter out examples whose next site falls outside the locus (so they have no label).
        """
        out: List[Dict[str, Any]] = []
        strand = g["strand"]
        tx_list = g.get("transcripts") or []

        for tx in tx_list:
            xs = list(map(int, tx["exon_starts"]))
            xe = list(map(int, tx["exon_ends"]))
            if not xs or not xe or len(xs) != len(xe):
                continue

            roles, pos = _splice_sites(strand, xs, xe)

            for i, r in enumerate(roles):
                if i + 1 >= len(roles):
                    continue

                ctx_abs = int(pos[i])
                nxt_abs = int(pos[i + 1])

                # ensure both context and next are inside the locus window
                # context must be inside [locus_start, locus_end)
                if not (locus_start <= ctx_abs < locus_end):
                    continue
                # next site must land in the *suffix* (strictly after context)
                if not (locus_start <= nxt_abs < locus_end):
                    continue

                out.append({
                    "context_pos": ctx_abs,
                    "next_pos": nxt_abs,
                })

        return out

    # --------- main iterator (batches of transcripts) ---------

    def __iter__(self):
        """
        Stream fixed-size transcript batches with carry-over between genes.
        """
        carry: List[Dict[str, Any]] = []
        carry_gene_keys: List[Any] = []

        for g in self._iter_genes():
            chrom, strand = g["chrom"], g["strand"]
            gs, ge = int(g["start"]), int(g["end"])
            s, e = self._locus_window(gs, ge, self.pad)
            if self.max_len is not None and (e - s) > self.max_len:
                continue

            txs = self._tx_examples(g, s, e)
            if not txs:
                continue

            gene_key = (g["gene_id"], chrom, strand, s, e)

            # enqueue this gene's transcripts
            for t in txs:
                carry.append(t)
                carry_gene_keys.append(gene_key)

            # emit as many full batches as possible
            while len(carry) >= self.tx_bs:
                batch_items = carry[: self.tx_bs]
                batch_gkeys = carry_gene_keys[: self.tx_bs]
                carry = carry[self.tx_bs :]
                carry_gene_keys = carry_gene_keys[self.tx_bs :]
                yield self._pack_batch(batch_items, batch_gkeys)

        # final partial batch (keep it; comment out to drop)
        if carry:
            yield self._pack_batch(carry, carry_gene_keys)

    @torch.no_grad()
    def _embed_locus(self, key):
        """
        Return a CPU tensor [L, D] of Caduceus embeddings over [s, e) on `chrom`.
        - No reverse-complement averaging.
        - Embeddings are aligned to forward genomic coordinates (index 0 -> s).
        """
        gene_id, chrom, strand, s, e = key
        # LRU cache
        if key in self._emb_cache:
            emb = self._emb_cache.pop(key)
            self._emb_cache[key] = emb
            return emb

        L = int(e) - int(s)
        if L <= 0:
            return torch.zeros(0, 256)

        # fetch sequence
        seq = str(self.human[chrom][s:e])

        if strand == "-":
            seq = reverse_complement(seq)

        # tokenize & run model
        tokens = self.tokenizer(seq, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model(**tokens, output_hidden_states=True)
        h = out.hidden_states[-1].squeeze(0)    # [L, D]

        if strand == "-":
            # Flip back so index 0 still corresponds to genomic s
            h = torch.flip(h, dims=[0])

        emb = h.detach().cpu()                  # keep CPU for downstream batching

        # tiny LRU (≤ batch size)
        self._emb_cache[key] = emb
        if len(self._emb_cache) > self.tx_bs:
            self._emb_cache.popitem(last=False)

        return emb

    # --------- batch assembly ---------
    def _pack_batch(
        self,
        items: List[Dict[str, Any]],
        gkeys: List[Any],
    ) -> Dict[str, Any]:
        """
        Assemble a minibatch for *next-site position* prediction:
        - X_context:  [B, D]  embedding at context site
        - X_suffix:   [B, S, D] embeddings from context+1 .. locus_end (padded)
        - suffix_mask:[B, S]
        - target_idx: [B] index in 0..S-1 of the next site within the suffix
        """
        B = len(items)

        # Compute per-item relative indices and suffix lengths
        ctx_rel_idx = torch.empty(B, dtype=torch.long)
        tgt_rel_idx = torch.empty(B, dtype=torch.long)
        suffix_lengths: List[int] = []

        for i, (it, key) in enumerate(zip(items, gkeys)):
            # key = (gene_id, chrom, strand, s, e)
            _, _, strand, s, e = key
            s, e = int(s), int(e)
            L = e - s

            ctx_rel = int(it["context_pos"]) - s
            nxt_rel = int(it["next_pos"]) - s

            if strand == "+":
                # suffix runs rightward
                tgt_idx = nxt_rel - (ctx_rel + 1)
                suffix_len = L - (ctx_rel + 1)
            else:
                # suffix runs leftward, toward smaller genomic coords
                tgt_idx = (ctx_rel - 1) - nxt_rel
                suffix_len = ctx_rel  # suffix length = number of bp leftward

            ctx_rel_idx[i] = ctx_rel
            tgt_rel_idx[i] = tgt_idx
            suffix_lengths.append(suffix_len)

        S = max(suffix_lengths)

        # Infer D
        D = 256

        X_context = torch.empty(B, D)
        X_suffix  = torch.zeros(B, S, D)
        suffix_mask = torch.zeros(B, S)
        target_idx = tgt_rel_idx.clone()  # [B]

        for i, (key, ctx_idx, slen) in enumerate(zip(gkeys, ctx_rel_idx.tolist(), suffix_lengths)):
            emb = self._embed_locus(key)  # [L, D]
            _, _, strand, s, e = key
            idx = int(ctx_idx)

            # context embedding
            X_context[i] = emb[idx]

            if slen > 0:
                if strand == "+":
                    seg = emb[idx + 1 : idx + 1 + slen]        # rightward
                else:
                    left = max(0, idx - slen)
                    seg = emb[left : idx]                       # leftward slice
                    seg = torch.flip(seg, dims=[0])             # make suffix[0] nearest to context
                X_suffix[i, :slen] = seg
                suffix_mask[i, :slen] = 1


        return {
            "X_context": X_context,       # [B, D]
            "X_suffix": X_suffix,         # [B, S, D]
            "suffix_mask": suffix_mask,   # [B, S]
            "target_idx": target_idx,     # [B], 0..S-1
        }
