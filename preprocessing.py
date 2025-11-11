import re
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path("data/raw_data")
OUT_DIR = Path("data/table")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_genome_assembly():
    df = pd.read_csv(RAW_DIR / "metadata.tsv", sep="\t")
    accession_to_assembly = dict(zip(df["File accession"], df["File assembly"]))
    return accession_to_assembly

def parse_attrs(s):
    return dict(re.findall(r'(\S+)\s+"([^"]+)"', s))

def gtf_to_parquet(gtf_path, out_parquet):
    # Load GTF and extract attributes
    cols = ["chrom","source","feature","start","end","score","strand","frame","attrs"]
    df = pd.read_csv(gtf_path, sep="\t", comment="#", names=cols, dtype={"chrom":"string"})
    ad = df["attrs"].map(parse_attrs)
    for k in ["gene_id","transcript_id","exon_number"]:
        df[k] = ad.map(lambda d: d.get(k))

    # Per-transcript table
    tx = (
        df[df.feature == "transcript"]
        [["chrom","gene_id","strand","transcript_id","start","end"]]
        .drop_duplicates()
        .rename(columns={"start":"tx_start","end":"tx_end"})
    )

    # Exon rows
    ex = df[df.feature == "exon"][["transcript_id","start","end","exon_number"]].copy()
    ex = ex[ex["transcript_id"].notna()]
    ex["exon_number"] = pd.to_numeric(ex["exon_number"], errors="coerce")

    ex_sorted = ex.sort_values(["transcript_id","exon_number","start","end"])

    # Aggregate exons
    exon_lists = (
        ex_sorted.groupby("transcript_id", sort=False)
        .agg(exon_starts=("start", list), exon_ends=("end", list))
        .reset_index()
    )

    # Attach exon lists to transcripts
    tx = tx.merge(exon_lists, on="transcript_id", how="left")
    tx = tx[tx["exon_starts"].notna()].reset_index(drop=True)

    # Basic alignment sanity: same list lengths
    same_len = tx.apply(lambda r: len(r["exon_starts"]) == len(r["exon_ends"]), axis=1)
    tx = tx[same_len].reset_index(drop=True)

    genes = (
        df[df.feature == "gene"]
        [["chrom","gene_id","strand","start","end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Pre-merge gene metadata into tx for O(1) access inside grouping
    tx = tx.merge(
        genes.rename(columns={"start":"gene_start","end":"gene_end", "strand":"gene_strand"}),
        on=["chrom","gene_id"],
        how="left",
        validate="many_to_one",
    )

    # If any transcripts lack a gene row, drop them (rare but safe)
    tx = tx[tx["gene_start"].notna()].reset_index(drop=True)

    # Build gene-centric rows (no redundant chrom/strand/gene_id inside transcripts)
    rows = []
    for (chrom, gid), gtx in tx.groupby(["chrom","gene_id"], sort=False):
        # gene meta is identical within group after the merge
        g_any = gtx.iloc[0]
        transcripts = []
        for r in gtx.itertuples(index=False):
            transcripts.append({
                "transcript_id": r.transcript_id,
                "tx_start": int(r.tx_start),
                "tx_end":   int(r.tx_end),
                "exon_starts": [int(x) for x in r.exon_starts],
                "exon_ends":   [int(x) for x in r.exon_ends],
            })
        if transcripts:
            rows.append({
                "gene_id": gid,
                "chrom": chrom,
                "start": int(g_any.gene_start),
                "end":   int(g_any.gene_end),
                "strand": g_any.gene_strand,
                "transcripts": transcripts,
            })

    # Arrow schema and write Parquet (one file, many genes)
    tx_struct = pa.struct([
        pa.field("transcript_id", pa.string()),
        pa.field("tx_start", pa.int64()),
        pa.field("tx_end", pa.int64()),
        pa.field("exon_starts", pa.list_(pa.int64())),
        pa.field("exon_ends", pa.list_(pa.int64())),
    ])
    schema = pa.schema([
        pa.field("gene_id", pa.string()),
        pa.field("chrom", pa.string()),
        pa.field("start", pa.int64()),
        pa.field("end", pa.int64()),
        pa.field("strand", pa.string()),
        pa.field("transcripts", pa.list_(tx_struct)),
    ])

    table = pa.Table.from_pydict(
        {
            "gene_id":     [r["gene_id"] for r in rows],
            "chrom":       [r["chrom"] for r in rows],
            "start":       [r["start"] for r in rows],
            "end":         [r["end"] for r in rows],
            "strand":      [r["strand"] for r in rows],
            "transcripts": [r["transcripts"] for r in rows],
        },
        schema=schema,
    )

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_parquet, compression="zstd", use_dictionary=True)

def worker(gtf_path: Path):
    accession = Path(gtf_path.stem).stem
    out_path = OUT_DIR / f"{accession}.parquet"
    if out_path.exists():
        return (str(gtf_path), f"{out_path} (skipped, already exists)")
    gtf_to_parquet(str(gtf_path), str(out_path))
    return (str(gtf_path), str(out_path))

def convert_all(in_dir: Path, pattern: str = "*.gtf.gz"):
    paths = sorted(in_dir.glob(pattern))
    print(f"Converting {len(paths)} files sequential...")

    ok, fail = 0, 0
    for p in tqdm(paths, desc="Processing GTFs"):
        try:
            src, dst = worker(p)
            ok += 1
            status = "SKIP" if "skipped" in dst else "OK"
            tqdm.write(f"[{status}] {src} -> {dst}")
        except Exception as e:
            fail += 1
            tqdm.write(f"[FAIL] {p}: {e}")

    print(f"Done. Success: {ok}, Failed: {fail}")

if __name__ == "__main__":
    convert_all(in_dir=RAW_DIR, pattern="*.gtf.gz")
