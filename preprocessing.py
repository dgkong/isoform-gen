import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

RAW_DIR = Path("data/raw_data")
OUT_DIR = Path("data/table")
METADATA_PATH = RAW_DIR / "metadata.tsv"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_file_mapping(metadata_path):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
    meta = pd.read_csv(metadata_path, sep="\t")
    
    gtfs = meta[
        (meta["File format"] == "gtf") & 
        (meta["Output type"] == "transcriptome annotations")
    ].copy()
    
    tsvs = meta[
        (meta["File format"] == "tsv") & 
        (meta["Output type"] == "transcript quantifications")
    ].copy()
    
    join_keys = ["Experiment accession", "Biological replicate(s)"]
    
    merged = pd.merge(
        gtfs[["File accession"] + join_keys],
        tsvs[["File accession"] + join_keys],
        on=join_keys,
        suffixes=("_gtf", "_tsv"),
        how="inner"
    )
    
    mapping = pd.Series(
        merged["File accession_tsv"].values, 
        index=merged["File accession_gtf"]
    ).to_dict()
    
    print(f"Mapped {len(mapping)} GTF-TSV pairs from metadata.")
    return mapping


def gtf_to_parquet(gtf_path, tsv_path, out_parquet):
    # Load GTF (Structure)
    cols = ["chrom", "feature", "start", "end", "strand", "attrs"]
    dtypes = {"chrom": "category", "feature": "category", "strand": "category"}
    
    df = pd.read_csv(
        gtf_path, 
        sep="\t", 
        comment="#", 
        names=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attrs"],
        usecols=cols,
        dtype=dtypes
    )

    spike_cats = df["chrom"].cat.categories[
        df["chrom"].cat.categories.str.contains("ERCC|Spikein", case=False, na=False)
    ]
    mask = df["chrom"].isin(spike_cats)
    df = df[~mask]

    df["gene_id"] = df["attrs"].str.extract(r'gene_id "([^"]+)"', expand=False)
    df["transcript_id"] = df["attrs"].str.extract(r'transcript_id "([^"]+)"', expand=False)
    df["exon_number"] = df["attrs"].str.extract(r'exon_number "([^"]+)"', expand=False)
    df.drop(columns=["attrs"], inplace=True)

    tx = (
        df[df.feature == "transcript"]
        [["chrom", "gene_id", "strand", "transcript_id", "start", "end"]]
        .drop_duplicates()
        .rename(columns={"start": "tx_start", "end": "tx_end"})
    )

    # Load TSV (Abundance)
    quant_df = pd.read_csv(tsv_path, sep="\t")
    
    abundance_col = quant_df.columns[-1]
    if not pd.api.types.is_numeric_dtype(quant_df[abundance_col]) or not abundance_col.startswith("rep"):
        raise ValueError(f"Invalid transcript abundance column: {abundance_col}")
    
    quant_df = quant_df[["annot_transcript_id", abundance_col]].copy()
    
    # Merge & Ratio Calculation
    tx = tx.merge(
        quant_df, 
        left_on="transcript_id", 
        right_on="annot_transcript_id", 
        how="left"
    )
    
    tx[abundance_col] = tx[abundance_col].fillna(0.0)

    tx["gene_total"] = tx.groupby("gene_id", observed=True)[abundance_col].transform("sum")
    tx["rel_abundance"] = tx[abundance_col] / tx["gene_total"]
    tx["rel_abundance"] = tx["rel_abundance"].fillna(0.0)
    
    # Process Exons
    ex = df[df.feature == "exon"].copy()
    ex = ex[["transcript_id", "start", "end", "exon_number"]]
    ex["exon_number"] = pd.to_numeric(ex["exon_number"], errors="coerce")
    
    ex_sorted = ex.sort_values(["transcript_id", "exon_number", "start", "end"])
    
    exon_lists = (
        ex_sorted.groupby("transcript_id", sort=False)
        .agg(exon_starts=("start", list), exon_ends=("end", list))
        .reset_index()
    )

    tx = tx.merge(exon_lists, on="transcript_id", how="left")
    tx = tx[tx["exon_starts"].notna()].reset_index(drop=True)
    
    lens_start = tx["exon_starts"].str.len()
    lens_end = tx["exon_ends"].str.len()
    tx = tx[lens_start == lens_end].reset_index(drop=True)

    # Gene-level aggregation
    genes = (
        df[df.feature == "gene"]
        [["chrom", "gene_id", "strand", "start", "end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    tx = tx.merge(
        genes.rename(columns={"start": "gene_start", "end": "gene_end", "strand": "gene_strand"}),
        on=["chrom", "gene_id"],
        how="left",
        validate="many_to_one",
    )
    
    tx = tx[tx["gene_start"].notna()].reset_index(drop=True)

    rows = []
    for (chrom, gid), gtx in tx.groupby(["chrom", "gene_id"], sort=False, observed=True):
        if gtx["gene_total"].iloc[0] == 0:
            continue
        g_any = gtx.iloc[0]
        transcripts = []
        for r in gtx.itertuples(index=False):
            if r.rel_abundance > 0.0:
                transcripts.append({
                    "transcript_id": r.transcript_id,
                    "tx_start": int(r.tx_start),
                    "tx_end":   int(r.tx_end),
                    "exon_starts": [int(x) for x in r.exon_starts],
                    "exon_ends":   [int(x) for x in r.exon_ends],
                    "rel_abundance": float(r.rel_abundance)
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

    tx_struct = pa.struct([
        pa.field("transcript_id", pa.string()),
        pa.field("tx_start", pa.int64()),
        pa.field("tx_end", pa.int64()),
        pa.field("exon_starts", pa.list_(pa.int64())),
        pa.field("exon_ends", pa.list_(pa.int64())),
        pa.field("rel_abundance", pa.float32()),
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


def worker(gtf_path: Path, file_mapping: dict):
    file_id = gtf_path.name.split('.')[0]
    out_path = OUT_DIR / f"{file_id}.parquet"
    
    if out_path.exists():
        return (str(gtf_path), f"{out_path} (skipped)")

    tsv_id = file_mapping.get(file_id)
    if not tsv_id:
        raise ValueError(f"No matching TSV found in metadata for GTF ID: {file_id}")

    tsv_path = RAW_DIR / f"{tsv_id}.tsv"
    if not tsv_path.exists():
        tsv_path_gz = RAW_DIR / f"{tsv_id}.tsv.gz"
        if tsv_path_gz.exists():
            tsv_path = tsv_path_gz
        else:
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    gtf_to_parquet(str(gtf_path), str(tsv_path), str(out_path))
    return (str(gtf_path), str(out_path))


def convert_sequential(in_dir: Path, metadata_path: Path, pattern: str = "*.gtf.gz"):
    mapping = build_file_mapping(metadata_path)
    paths = sorted(in_dir.glob(pattern))
 
    ok, fail = 0, 0
    for p in tqdm(paths, desc="Processing GTFs"):
        try:
            src, dst = worker(p, mapping)
            ok += 1
            status = "SKIP" if "skipped" in dst else "OK"
            tqdm.write(f"[{status}] {src} -> {dst}")
        except Exception as e:
            fail += 1
            tqdm.write(f"[FAIL] {p.name}: {e}")

    print(f"Done. Success: {ok}, Failed: {fail}")


if __name__ == "__main__":
    convert_sequential(in_dir=RAW_DIR, metadata_path=METADATA_PATH, pattern="*.gtf.gz")
