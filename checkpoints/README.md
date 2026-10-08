# Checkpoints

Release checkpoints are self-describing: each file holds the model weights (`state_dict`), the
architecture they need (`dig2_config`), a format version, and the SHA-256 of the state_dict.
`python -m dig2.train +experiment=eval` loads `checkpoints/dig2.ckpt` by default and rebuilds the
model from the embedded architecture, so a checkpoint can never be paired with the wrong config.

| File | Model | Used for |
|---|---|---|
| `dig2.ckpt` | **Final DiG² model** — *weights to be added at release* | Main-paper results: `+experiment=eval` |

Place the files in this directory (they are git-ignored). Verify a download with

```bash
python -c "import torch; from dig2.checkpoint import state_dict_sha256 as h; \
c = torch.load('checkpoints/dig2.ckpt', map_location='cpu', weights_only=False); \
print(h(c['state_dict']) == c['state_dict_sha256'])"
```

Any checkpoint written by `dig2.train` can be turned into this format with
`python -m dig2.tools.export_ckpt <lightning.ckpt> <out.ckpt> [+experiment=...]`.
