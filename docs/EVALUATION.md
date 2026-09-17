# Evaluation

`evaluation/run.py` launches upstream evaluation code and saves its configuration,
code revision, raw predictions, and scores. It does not replace benchmark prompts
or scoring functions with training rewards. Use a separate evaluation environment.

## Multi-image and VSI benchmarks

Install [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and its Qwen-VL
dependencies following the upstream installation instructions. The checked
interface is revision `c44bc601dcf4698e3cf2fc851a6a0b00b13a2d06`:

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
git -C VLMEvalKit checkout c44bc601dcf4698e3cf2fc851a6a0b00b13a2d06
python -m pip install -e ./VLMEvalKit
```

Run from this repository, using the Python environment in which VLMEvalKit is installed:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py \
  --bench vsi --family qwen25vl \
  --model kagakouko/Spatial-Interactor-Qwen2.5-VL-7B \
  --toolkit ./VLMEvalKit --data-root /data/benchmarks \
  --output ./outputs/qwen25vl7b/vsi
```

Use `--family qwen3vl` for either Qwen3-VL checkpoint. Supported benchmark names:

| Argument | Upstream task |
|---|---|
| `vsi` | `VSI-Bench`, 32 frames by default |
| `mindcube` | `MindCubeBench_tiny_raw_qa` |
| `spbench` | `SPBench-MV` |
| `mmsi` | `MMSIBench_wo_circular` |
| `viewspatial` | `ViewSpatialBench` |

The launcher uses direct-answer prompts supplied by the dataset, greedy decoding,
a 64-token response budget, and `exact_matching` judging. Image pixel budgets,
video frame count, and response length are explicit command-line options. These
are release evaluation defaults, not a claim that every historical run used this
exact toolkit revision. Keep settings identical across checkpoints being compared.

`--dry-run` prints the command and configuration without loading a model.
`--resume` reuses predictions only if the saved settings and code revision match.
For scoring existing predictions, use `--mode eval --resume`.

## VSTI and SAT

These tasks use the authors' **lmms-eval task definitions** instead of a local
reimplementation. Obtain the datasets and task files from:

- [VSTI / VLM-3R](https://github.com/VITA-Group/VLM-3R/tree/main/thinking-in-space/lmms_eval/tasks/vstibench)
- [SAT author evaluation repository](https://github.com/arijitray1993/lmms-eval)

Install an lmms-eval checkout supporting your backbone. Older author forks may
not include `qwen3_vl`; use the author task definitions through `--include-path`
with a compatible [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
checkout. Prepare the author YAML/media paths before launching; the launcher
does not download or repair those task definitions automatically.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run.py \
  --bench vsti --family qwen3vl \
  --model kagakouko/Spatial-Interactor-Qwen3-VL-8B \
  --toolkit ./lmms-eval --include-path /data/author_tasks \
  --output ./outputs/qwen3vl8b/vsti
```

For SAT, use `--bench sat-real` or `--bench sat-syn`, with the respective
`sat_real` / `sat_syn` task available in that checkout or include directory.
The task YAML controls the response budget unless `--max-new-tokens` is set.
VSTI uses the official numerical MRA and option scoring, not a strict 10% metric.
Pin the VSTI dataset revision: its annotations have changed, so old and corrected
labels must not be mixed in one comparison. For SAT, retain the author's answer
order processing and aggregation; record the evaluated split and resulting count.

All runs cover the selected task split without result-based filtering. Preserve
`run_manifest.json` and the `results/` directory for inspecting individual failures.
The launcher has CPU-side configuration tests; GPU execution still depends on
the installed backend, prepared benchmark media, and available hardware.
