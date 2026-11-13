# Evaluation

The training runner renders held-out views at arrival and re-renders every stored visit after each update. It checks exact history, input identities and finite PSNR, SSIM and LPIPS. The scripts below validate those outputs and aggregate them; they do not substitute a render of the latest physical state for an earlier stored visit.

```bash
python eval/multistate.py --subset synthetic
python eval/multistate.py --subset real
```

If preparation used a scene subset, pass the same ordered `--scenes` list. Reports distinguish initial-included averages from later-only averages, weight visits equally within scenes and scenes equally, and count the initial model once in raw history storage. Full history retrieval and current-view RGB access checks must pass before interpreting arrival and final archival recall as equal.
