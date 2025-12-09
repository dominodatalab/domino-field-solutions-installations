#!/usr/bin/env python3
import argparse, os, io, time, math, csv, hashlib, random, string
from statistics import mean, median
from typing import Dict, List, Tuple

# ------------------------
# Utilities
# ------------------------
def rand_name(prefix: str, length=8):
    suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    return f"{prefix}{suffix}"

def pattern_gen(total_bytes: int, chunk_bytes: int):
    # deterministic repeating pattern
    buf = (b"\xAB" * chunk_bytes)[:chunk_bytes]
    remaining = total_bytes
    while remaining > 0:
        n = min(remaining, chunk_bytes)
        yield buf[:n]
        remaining -= n

def now(): return time.perf_counter()

def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if abs(n) < 1024.0:
            return f"{n:3.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} PB"

def pct(v, p):
    if not v: return float("nan")
    idx = max(0, min(len(v)-1, math.ceil(p/100 * len(v)) - 1))
    return sorted(v)[idx]

def md5_update(h, chunk: bytes):
    h.update(chunk)

# ------------------------
# Backends
# ------------------------
class BackendBase:
    def write_stream(self, key: str, total_bytes: int, chunk_bytes: int) -> Tuple[int, str]:
        """Stream write; returns (bytes_written, md5_hex)."""
        raise NotImplementedError
    def read_to_local(self, key: str, dst_path: str, chunk_bytes: int) -> Tuple[int, str]:
        """Stream read to local dst; returns (bytes_read, md5_hex_of_stream)."""
        raise NotImplementedError
    def delete_key(self, key: str): ...
    def join(self, prefix: str, name: str): ...

class Boto3Backend(BackendBase):
    def __init__(self, bucket, prefix, region=None, multipart_chunksize=8*1024*1024, max_concurrency=16):
        import boto3
        from botocore.config import Config
        from boto3.s3.transfer import TransferConfig

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.multipart_chunksize = multipart_chunksize

        cfg = Config(
            region_name=region,
            max_pool_connections=max(max_concurrency * 2, 32),
            retries={"max_attempts": 10, "mode": "adaptive"},
        )
        self.boto3 = boto3
        self.s3c = boto3.client("s3", config=cfg)
        self.TransferConfig = TransferConfig
        self.transfer_cfg = TransferConfig(
            multipart_threshold=8*1024*1024,
            multipart_chunksize=multipart_chunksize,
            max_concurrency=max_concurrency,
            use_threads=True,
        )

    def _k(self, key: str) -> str:
        return f"{self.prefix}/{key}".lstrip("/") if self.prefix else key

    # streaming upload using file-like wrapper
    class _Stream(io.RawIOBase):
        def __init__(self, total, chunk, on_chunk):
            self._gen = pattern_gen(total, chunk)
            self._on = on_chunk
        def readable(self): return True
        def readinto(self, b):
            try:
                chunk = next(self._gen)
            except StopIteration:
                return 0
            n = len(chunk)
            b[:n] = chunk
            if self._on: self._on(chunk)
            return n

    def write_stream(self, key: str, total_bytes: int, chunk_bytes: int) -> Tuple[int, str]:
        h = hashlib.md5()
        stream = Boto3Backend._Stream(total_bytes, chunk_bytes, lambda c: md5_update(h, c))
        t0 = now()
        self.s3c.upload_fileobj(stream, self.bucket, self._k(key), Config=self.transfer_cfg)
        _ = now() - t0
        return total_bytes, h.hexdigest()

    def read_to_local(self, key: str, dst_path: str, chunk_bytes: int) -> Tuple[int, str]:
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        resp = self.s3c.get_object(Bucket=self.bucket, Key=self._k(key))
        h = hashlib.md5()
        total = 0
        with open(dst_path, "wb", buffering=0) as out:
            body = resp["Body"]
            while True:
                buf = body.read(chunk_bytes)
                if not buf:
                    break
                out.write(buf)
                h.update(buf)
                total += len(buf)
            out.flush()
            os.fsync(out.fileno())
        return total, h.hexdigest()

    def delete_key(self, key: str):
        self.s3c.delete_object(Bucket=self.bucket, Key=self._k(key))

    def join(self, prefix: str, name: str):
        return f"{prefix.rstrip('/')}/{name}"

class PosixBackend(BackendBase):
    """Generic POSIX (used for FUSE and EFS)"""
    def __init__(self, root_path, prefix):
        self.root = root_path.rstrip("/")
        self.prefix = prefix.strip("/")

    def _path(self, key):
        rel = key if not self.prefix else f"{self.prefix}/{key}"
        p = os.path.join(self.root, rel)
        d = os.path.dirname(p)
        os.makedirs(d, exist_ok=True)
        return p

    def write_stream(self, key: str, total_bytes: int, chunk_bytes: int) -> Tuple[int, str]:
        p = self._path(key)
        h = hashlib.md5()
        written = 0
        with open(p, "wb", buffering=0) as f:
            for chunk in pattern_gen(total_bytes, chunk_bytes):
                f.write(chunk)
                h.update(chunk)
                written += len(chunk)
            f.flush()
            os.fsync(f.fileno())
        return written, h.hexdigest()

    def read_to_local(self, key: str, dst_path: str, chunk_bytes: int) -> Tuple[int, str]:
        src = self._path(key)
        os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
        h = hashlib.md5()
        total = 0
        with open(src, "rb", buffering=0) as fin, open(dst_path, "wb", buffering=0) as out:
            while True:
                buf = fin.read(chunk_bytes)
                if not buf: break
                out.write(buf)
                h.update(buf)
                total += len(buf)
            out.flush()
            os.fsync(out.fileno())
        return total, h.hexdigest()

    def delete_key(self, key: str):
        p = self._path(key)
        try: os.remove(p)
        except FileNotFoundError: pass

    def join(self, prefix: str, name: str):
        return f"{prefix.rstrip('/')}/{name}"

# ------------------------
# Bench logic
# ------------------------
def bench_writes(backend: BackendBase, label: str, base_prefix: str, files_n: int, file_mb: int, chunk_mb: int, verify: bool):
    keys = [backend.join(base_prefix, f"obj_{i:03d}_{file_mb}MB.bin") for i in range(files_n)]
    size_per = file_mb * 1024 * 1024
    chunk = chunk_mb * 1024 * 1024
    elapsed = []
    md5s: Dict[str, str] = {}
    total_bytes = 0

    print(f"[{label}] Writing {files_n} files x {file_mb} MB (chunk {chunk_mb} MB)...")
    for k in keys:
        t0 = now()
        n, md5hex = backend.write_stream(k, size_per, chunk)
        dt = now() - t0
        elapsed.append(dt)
        total_bytes += n
        if verify: md5s[k] = md5hex

    tp = (total_bytes / (1024*1024)) / sum(elapsed) if elapsed else 0.0
    stats = {
        "label": label,
        "phase": "write",
        "files": files_n,
        "bytes": total_bytes,
        "MBps": tp,
        "p50_ms": median(elapsed)*1000 if elapsed else float("nan"),
        "p95_ms": pct(elapsed, 95)*1000 if elapsed else float("nan"),
        "p99_ms": pct(elapsed, 99)*1000 if elapsed else float("nan"),
    }
    return keys, md5s, stats

def bench_read_copy_delete(backend: BackendBase, label: str, keys: List[str], local_dir: str, chunk_mb: int, expected_md5: Dict[str,str], verify: bool):
    os.makedirs(local_dir, exist_ok=True)
    elapsed = []
    total_bytes = 0
    mismatches = 0
    chunk = chunk_mb * 1024 * 1024

    print(f"[{label}] Read→copy→delete-local for {len(keys)} files (chunk {chunk_mb} MB)...")
    for k in keys:
        local_path = os.path.join(local_dir, os.path.basename(k))
        t0 = now()
        n, md5hex = backend.read_to_local(k, local_path, chunk)
        # delete local before next file (per requirement)
        try: os.remove(local_path)
        except FileNotFoundError: pass
        dt = now() - t0
        elapsed.append(dt)
        total_bytes += n
        if verify and expected_md5:
            exp = expected_md5.get(k)
            if exp and md5hex != exp:
                mismatches += 1

    tp = (total_bytes / (1024*1024)) / sum(elapsed) if elapsed else 0.0
    stats = {
        "label": label,
        "phase": "read_copy_delete",
        "files": len(keys),
        "bytes": total_bytes,
        "MBps": tp,
        "p50_ms": median(elapsed)*1000 if elapsed else float("nan"),
        "p95_ms": pct(elapsed, 95)*1000 if elapsed else float("nan"),
        "p99_ms": pct(elapsed, 99)*1000 if elapsed else float("nan"),
        "md5_mismatch_files": mismatches if verify else 0,
    }
    return stats

def print_compare_table(results):
    # results: list of stats dicts
    writes = [r for r in results if r["phase"] == "write"]
    reads  = [r for r in results if r["phase"] == "read_copy_delete"]

    print("\n=== Phase A: Writes (higher MB/s is better) ===")
    print(f"{'Backend':<12} {'Files':>5} {'Size':>10} {'MB/s':>10} {'P50 ms':>10} {'P95 ms':>10} {'P99 ms':>10}")
    for r in writes:
        print(f"{r['label']:<12} {r['files']:>5} {human(r['bytes']):>10} {r['MBps']:>10.1f} {r['p50_ms']:>10.1f} {r['p95_ms']:>10.1f} {r['p99_ms']:>10.1f}")

    print("\n=== Phase B: Read → Copy Local → Delete Local (higher MB/s is better) ===")
    print(f"{'Backend':<12} {'Files':>5} {'Size':>10} {'MB/s':>10} {'P50 ms':>10} {'P95 ms':>10} {'P99 ms':>10} {'MD5 mism':>10}")
    for r in reads:
        print(f"{r['label']:<12} {r['files']:>5} {human(r['bytes']):>10} {r['MBps']:>10.1f} {r['p50_ms']:>10.1f} {r['p95_ms']:>10.1f} {r['p99_ms']:>10.1f} {r.get('md5_mismatch_files',0):>10}")

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Benchmark S3 (boto3) vs FUSE vs EFS: write, then read→copy→delete-local.")
    # boto3 args
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix-s3", default="bench/s3")
    ap.add_argument("--region", default=None)
    ap.add_argument("--multipart-chunk-mb", type=int, default=8)
    ap.add_argument("--max-concurrency", type=int, default=24)

    # fuse args
    ap.add_argument("--mount-path", required=True, help="S3 fuse mount path (e.g., /mnt/s3)")
    ap.add_argument("--prefix-fuse", default="bench/fuse")

    # efs args
    ap.add_argument("--efs-path", required=True, help="EFS mount path (e.g., /mnt/efs)")
    ap.add_argument("--prefix-efs", default="bench/efs")

    # workload
    ap.add_argument("--files-n", type=int, default=8, help="Number of files to write/read")
    ap.add_argument("--file-mb", type=int, default=256, help="Size per file (MB)")
    ap.add_argument("--chunk-mb", type=int, default=8, help="I/O chunk size (MB)")
    ap.add_argument("--local-dir", default="/mnt/data/tmp_copy", help="Where to place local copies during read tests")
    ap.add_argument("--verify", action="store_true", help="Compute + compare MD5s")

    # csv
    ap.add_argument("--out-csv", default="s3_fuse_efs_results.csv")

    args = ap.parse_args()

    # Instantiate backends
    s3 = Boto3Backend(
        bucket=args.bucket,
        prefix=args.prefix_s3,
        region=args.region,
        multipart_chunksize=args.multipart_chunk_mb * 1024 * 1024,
        max_concurrency=args.max_concurrency,
    )
    fuse = PosixBackend(root_path=args.mount_path, prefix=args.prefix_fuse)
    efs  = PosixBackend(root_path=args.efs_path,   prefix=args.prefix_efs)

    # File names are identical across backends (separate prefixes)
    base_names = [f"obj_{i:03d}_{args.file_m_b if hasattr(args,'file_m_b') else args.file_mb}MB.bin" for i in range(args.files_n)]
    s3_keys   = [s3.join("", n)   for n in base_names]
    fuse_keys = [fuse.join("", n) for n in base_names]
    efs_keys  = [efs.join("", n)  for n in base_names]

    results = []
    written = {}

    # Phase A: writes
    s3_written_keys, s3_md5, write_s3 = bench_writes(s3,  "S3(boto3)", "", args.files_n, args.file_mb, args.chunk_mb, args.verify)
    results.append(write_s3); written["s3"]= (s3, s3_written_keys, s3_md5)

    fuse_written_keys, fuse_md5, write_fuse = bench_writes(fuse, "FUSE",     "", args.files_n, args.file_mb, args.chunk_mb, args.verify)
    results.append(write_fuse); written["fuse"]= (fuse, fuse_written_keys, fuse_md5)

    efs_written_keys, efs_md5, write_efs = bench_writes(efs,  "EFS",      "", args.files_n, args.file_mb, args.chunk_mb, args.verify)
    results.append(write_efs); written["efs"]= (efs, efs_written_keys, efs_md5)

    # Phase B: read → local → delete local
    read_s3   = bench_read_copy_delete(s3,   "S3(boto3)", written["s3"][1],  args.local_dir, args.chunk_mb,  written["s3"][2],   args.verify)
    read_fuse = bench_read_copy_delete(fuse, "FUSE",      written["fuse"][1],args.local_dir, args.chunk_mb,  written["fuse"][2], args.verify)
    read_efs  = bench_read_copy_delete(efs,  "EFS",       written["efs"][1], args.local_dir, args.chunk_mb,  written["efs"][2],  args.verify)

    results.extend([read_s3, read_fuse, read_efs])

    # Print compare tables
    print_compare_table(results)

    # Append to CSV
    rows = []
    for r in results:
        rows.append({
            "label": r["label"],
            "phase": r["phase"],
            "files": r["files"],
            "bytes": r["bytes"],
            "MBps": f"{r['MBps']:.2f}",
            "p50_ms": f"{r['p50_ms']:.1f}",
            "p95_ms": f"{r['p95_ms']:.1f}",
            "p99_ms": f"{r['p99_ms']:.1f}",
            "md5_mismatch_files": r.get("md5_mismatch_files", 0),
        })

    exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    # Cleanup remote files (comment out if you want to keep them)
    for k in written["s3"][1]:   s3.delete_key(k)
    for k in written["fuse"][1]: fuse.delete_key(k)
    for k in written["efs"][1]:  efs.delete_key(k)

if __name__ == "__main__":
    main()

'''
python benchmark_s3_fuse_efs.py \
  --bucket wgamage \
  --prefix-s3 teams/team-c/bench-s3 \
  --region us-west-2 \
  --mount-path /domino/edv/mountpoints3-team-c-pvc/ \
  --prefix-fuse teams/team-c/bench-fuse \
  --efs-path /domino/datasets/local/quick-start \
  --prefix-efs teams/team-c/bench-efs \
  --files-n 8 \
  --file-mb 256 \
  --chunk-mb 8 \
  --verify \
  --out-csv triple_results.csv
'''