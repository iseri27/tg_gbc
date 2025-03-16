We provide an example for OPEN.

# 1. Install StreamPETR

Follow [StreamPETR](https://github.com/exiawsh/StreamPETR) to install StreamPETR, prepare datasets and download checkpoints.

# 2. Install tgGBC

```bash
git clone https://github.com/iseri27/tg_gbc
cd tg_gbc
python setup.py develop
```

# 3. Replace Files

```bash
mkdir -p $StreamPETR/projects/configs/tgtg
mkdir -p $StreamPETR/projects/mmdet3d_plugin/core/hook

cd tg_gbc
cp examples/streampetr/tools/test.py $StreamPETR/tools/
cp examples/streampetr/projects/mmdet3d_plugin/models/utils/petr_transformer.py $StreamPETR/projects/mmdet3d_plugin/models/utils/
cp examples/streampetr/projects/mmdet3d_plugin/models/dense_heads/streampetr_head.py $StreamPETR/projects/mmdet3d_plugin/models/dense_heads/
cp examples/streampetr/projects/configs/tgtg/*.py $StreamPETR/projects/configs/tgtg/
cp examples/streampetr/projects/mmdet3d_plugin/core/hook/gbc_hook.py $StreamPETR/projects/mmdet3d_plugin/core/hook/
cp examples/streampetr/projects/mmdet3d_plugin/__init__.py $StreamPETR/projects/mmdet3d_plugin/
```

Note: `$StreamPETR` is the place where you install the OPEN project.

# 4. Train StreamPETR with tgGBC

```bash
cd $StreamPETR
# Train StreamPETR wiht tgGBC using a single GPU.
PYTHONPATH=$(pwd) python tools/train.py \
   projects/configs/tgtg/streampetr_vov_1600x640_bs2x2_24e_gbc_r21000_n1_k175.py

# Train StreamPETR wiht tgGBC using multi gpus.
export CUDA_VISIBLE_DEVICES="0,1"
PYTHONPATH=$(pwd) python -m torch.distributed.launch \
    --nproc_per_node 2 \
    --master_port 12000 \
    tools/train.py \
    projects/configs/tgtg/streampetr_vov_1600x640_bs2x2_24e_gbc_r21000_n1_k175.py \
    --launcher pytorch
```
To train models with tgGBC, there are three stages: warmup, heating and pruning.

- `warmup`: training w/o tgGBC
- `heating`: training w/ growing $r$
- `pruning`: training w/ maximum $r$

To change the ratio of warmup/heating/pruning, please modify `custom_hooks` in your config file like:

```python
custom_hooks = [
    dict(
        type="GBCHook",
            warmup_iters=1 * num_iters_per_epoch, # warmup for 1 epoch
            heating_iters=2 * num_iters_per_epoch, # heating for 2 epochs
            r=21000,
            n=1,
            k=175,
            layers=6,
            log_interval=50,
    )
]
```

The settings for `r`, `n`, `k`, and `layers` in `custom_hooks` must be the same as those in the `model.pts_bbox_head.transformer.decoder`.