# Render a stored visit

After training a ReScene sequence, re-render an observed visit at its official held-out cameras:

```bash
python demo/render_visit.py --subset real \
  --run outputs/rescene-real-reproduction/rescene/can/full --visit 0 --gpu 0
```

Use `--subset synthetic` with the corresponding synthetic run directory. CLI visit indices are zero-based. This command uses the prepared dataset, original worker and source guards, requires CUDA, and writes normal evaluation renders and metrics. Existing renderings are checked for exact agreement by the worker. Do not run it concurrently with a training process writing the same output directory.

This is a local reconstruction utility. The promotional project website, videos and comparison tables live in the separate `3DGSVer` GitHub Pages repository.
