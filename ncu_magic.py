# NVTX-only Nsight Compute cell-magic, no main() required.
# - wraps THIS CELL into a function (__user_code__())
# - runs it twice in one process: WARM then PROF
# - profiles only NVTX push/pop ranges named "PROF:<label>/" via --nvtx-include
# - outputs a tidy pandas table with pretty names & derived hit rates

import sys, shutil, tempfile, subprocess, textwrap
from pathlib import Path
from io import StringIO

import pandas as pd
from IPython.core.magic import register_cell_magic

# ---- Metrics: 7 core + 2 extras (L2 & DRAM % peak) ----
DEFAULT_METRICS = ",".join([
    # timing
    "gpu__time_duration.sum",
    # DRAM bytes
    "dram__bytes_read.sum","dram__bytes_write.sum",
    # L2 traffic & quality
    "lts__t_sectors_op_read.sum","lts__t_sectors_op_read_hit.sum","lts__t_sectors_evict.sum",
    # L1 load quality
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    # % of peak
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
])

# raw metric -> (pretty_name, scale, unit)
PRETTY = {
    "gpu__time_duration.sum": ("time_ms", 1e-6, "ms"),
    "dram__bytes_read.sum":   ("dram_read_GB", 1/1e9, "GB"),
    "dram__bytes_write.sum":  ("dram_write_GB", 1/1e9, "GB"),
    "lts__t_sectors_op_read.sum":      ("l2_read_GB", 32/1e9, "GB"),
    "lts__t_sectors_op_read_hit.sum":  ("_l2_read_hit_GB", 32/1e9, "GB"),  # temp for rate
    "lts__t_sectors_evict.sum":        ("l2_evicted_GB", 32/1e9, "GB"),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum":  ("_l1_hit", 1.0, ""),
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum": ("_l1_miss", 1.0, ""),
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": ("l2_pct_of_peak", 1.0, "%"),
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("dram_pct_of_peak", 1.0, "%"),
}

DISPLAY_COLS = [
    "time_ms",
    "dram_read_GB","dram_write_GB",
    "l2_read_GB","l2_read_hit_rate","l2_evicted_GB",
    "l1_glb_load_hit_rate",
    "l2_pct_of_peak","dram_pct_of_peak",
]

def _need(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise RuntimeError(f"Missing `{bin_name}` in PATH")
    return p

@register_cell_magic
def ncu_profile(line, cell):
    """
    %%ncu_profile --labels add_baseline,add_evict_first [--metrics m1,m2,...]
      Write kernels + launches inline; end your cell with nvtx_launch(...) calls.
      No main() needed. Returns a tidy pandas DataFrame.
    """
    toks = line.split()
    metrics = DEFAULT_METRICS
    labels_arg = None
    if "--metrics" in toks:
        i = toks.index("--metrics"); metrics = toks[i+1]
    if "--labels" in toks:
        i = toks.index("--labels"); labels_arg = toks[i+1]
    if not labels_arg:
        raise ValueError("Please pass --labels with comma-separated labels (e.g., add_baseline,add_evict_first).")

    labels = [s.strip() for s in labels_arg.split(",") if s.strip()]
    if not labels:
        raise ValueError("No labels parsed from --labels.")

    ncu = _need("ncu")

    # Build script: prolog (nvtx helper), wrapped user code, epilog (warm then prof)
    tmpdir = Path(tempfile.mkdtemp(prefix="ncu_cell_"))
    script = tmpdir / "cell_profile.py"

    prolog = textwrap.dedent("""
        import time, torch, triton, triton.language as tl
        torch.set_default_device("cuda")
        PROFILE_PHASE = "WARM"  # toggled to "PROF" before 2nd run

        def nvtx_launch(label, fn):
            prefix = "PROF:" if PROFILE_PHASE == "PROF" else "WARM:"
            torch.cuda.nvtx.range_push(prefix + label)
            fn()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    """)

    # indent the cell into a function body
    indented = textwrap.indent(cell.rstrip("\n"), "    ")
    wrapped = f"def __user_code__():\n{indented}\n"

    epilog = textwrap.dedent("""
        if __name__ == "__main__":
            # 1) Warmup pass
            PROFILE_PHASE = "WARM"
            __user_code__(); torch.cuda.synchronize(); time.sleep(0.02)
            # 2) Profile pass
            PROFILE_PHASE = "PROF"
            __user_code__(); torch.cuda.synchronize(); time.sleep(0.02)
    """)

    script.write_text(prolog + "\n# === USER CELL START ===\n" + wrapped + "\n# === USER CELL END ===\n" + epilog,
                      encoding="utf-8")

    # ncu command: capture only our PROF:.../ ranges
    cmd = [
        "sudo",
        _need("ncu"),
        "--csv",
        "--target-processes","all",
        "--set","full",
        "--nvtx",
    ]
    for lab in labels:
        cmd += ["--nvtx-include", f"PROF:{lab}/"]
    cmd += ["--metrics", metrics, "--", sys.executable, str(script)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout[:4000])
        raise RuntimeError("ncu failed; see output above.")

    # Extract the "Command line profiler metrics" CSV block
    lines = proc.stdout.splitlines()
    start = next((i for i,l in enumerate(lines) if "Kernel Name" in l and "Metric Name" in l), None)
    if start is None:
        print(proc.stdout[:2000])
        raise RuntimeError("Could not find metrics CSV in Nsight output.")
    block = []
    for j in range(start, len(lines)):
        if j > start and not lines[j].strip():
            break
        block.append(lines[j])

    df_long = pd.read_csv(StringIO("\n".join(block)))
    if "Section Name" in df_long.columns:
        df_long = df_long[df_long["Section Name"] == "Command line profiler metrics"]

    # Pivot: Kernel Name x Metric Name -> Metric Value
    tidy = df_long.pivot_table(index="Kernel Name", columns="Metric Name",
                               values="Metric Value", aggfunc="first").copy()

    # Convert to numeric where possible
    tidy = tidy.apply(pd.to_numeric, errors="coerce")
    tidy.index.name = None
    tidy.columns.name = None

    # Apply pretty mapping & scaling
    for raw, (nice, scale, _) in PRETTY.items():
        if raw in tidy.columns:
            tidy[nice] = tidy[raw] * scale

    # Derived rates
    if {"_l2_read_hit_GB","l2_read_GB"}.issubset(tidy.columns):
        denom = tidy["l2_read_GB"].replace(0, float("nan"))
        tidy["l2_read_hit_rate"] = (tidy["_l2_read_hit_GB"] / denom).clip(0, 1)

    if {"_l1_hit","_l1_miss"}.issubset(tidy.columns):
        denom = (tidy["_l1_hit"] + tidy["_l1_miss"]).replace(0, float("nan"))
        tidy["l1_glb_load_hit_rate"] = (tidy["_l1_hit"] / denom).clip(0, 1)

    # Keep compact tutorial columns (only those present)
    cols = [c for c in [
        "time_ms",
        "dram_read_GB","dram_write_GB",
        "l2_read_GB","l2_read_hit_rate","l2_evicted_GB",
        "l1_glb_load_hit_rate",
        "l2_pct_of_peak","dram_pct_of_peak",
    ] if c in tidy.columns]
    if cols:
        tidy = tidy[cols].sort_index()

    return tidy
