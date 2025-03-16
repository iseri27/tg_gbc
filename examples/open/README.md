We provide an example for OPEN.

# 1. Install OPEN

Follow [AlmoonYsl/OPEN](https://github.com/AlmoonYsl/OPEN) to install OPEN, prepare datasets and download checkpoints.

# 2. Install tgGBC

```bash
git clone https://github.com/iseri27/tg_gbc
cd tg_gbc
python setup.py develop
```

# 3. Replace Files

```bash
cd tg_gbc
cp examples/open/tools/test.py $OPEN/tools/
cp examples/open/projects/mmdet3d_plugin/models/utils/petr_transformer.py $OPEN/projects/mmdet3d_plugin/models/utils/
cp examples/open/projects/mmdet3d_plugin/models/dense_heads/open_head.py $OPEN/projects/mmdet3d_plugin/models/dense_heads/
```

Note: `$OPEN` is the place where you install the OPEN project.

# 4. Evaluate OPEN with tgGBC

```bash
cd $OPEN
# Evaluate OPEN wiht tgGBC using a single GPU.
PYTHONPATH=$(pwd) python tools/test.py \
    projects/configs/open_r101_1408_90e.py \
    ckpts/open_r101_1408x512_90e.pth \
    --eval bbox \
    --gbc -r 12000 -n 1

# Evaluate OPEN wiht tgGBC using multi gpus.
export CUDA_VISIBLE_DEVICES="0,1"
PYTHONPATH=$(pwd) python -m torch.distributed.launch \
    --nproc_per_node 2 \
    --master_port 12000 \
    tools/test.py \
    projects/configs/open_r101_1408_90e.py \
    ckpts/open_r101_1408x512_90e.pth \
    --eval bbox \
    --launcher pytorch \
    --gbc -r 12000 -n 1
```