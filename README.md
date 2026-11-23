# LoRA Adaptation Lab

A compact, reproducible NumPy experiment showing how a low-rank adapter learns a task-specific weight update while a base linear layer stays frozen. It reports held-out error, trainable-parameter savings, and the numerical equivalence of applying versus merging the adapter.

## Run it

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
ruff check .
lora-adaptation-lab
```

The CLI uses a fixed seed and generated matrices; `--seed` and `--steps` control the experiment. Output is JSON for easy inspection or reuse.

## What it demonstrates

The update is `ΔW = (alpha / rank) · A · B`. Only `A` and `B` are fitted with full-batch gradient descent; the base matrix is unchanged. The demo creates a known rank-2 target residual, learns it from synthetic training examples, and measures performance on a separate synthetic holdout. It also verifies that `X @ (W + ΔW)` equals the merged-weight route `X @ merge(W)`.

For a matrix with `d_in × d_out` weights and rank `r`, the adapter has `r(d_in + d_out)` trainable values instead of `d_in d_out`. This count is a parameter-count comparison, not a universal speed, memory, or quality guarantee.

## Scope

This is an educational linear-layer experiment, not an implementation of transformer fine-tuning, a pretrained language model, or a production training pipeline. Results are synthetic and say nothing about model quality on real tasks. The small NumPy implementation favors clarity and reproducibility over optimized training.

## License

MIT; see [LICENSE](LICENSE).
