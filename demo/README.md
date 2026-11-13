# Render a stored visit

After training, render a visit at its held-out cameras:

```bash
python demo/render_visit.py --subset real \
  --run outputs/rescene-real-photo-v1/rescene/can/full --visit 0 --gpu 0
```

Visit indices start at zero. For synthetic scenes, use `--subset synthetic` and the corresponding run directory under `outputs/rescene-synthetic-photo-v1/`.

The command recovers the visit from the stored updates and writes evaluation renders and metrics. Run it after training has finished.
