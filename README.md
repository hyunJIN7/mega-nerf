# Mega-NeRF

This repository contains the code needed to train [Mega-NeRF](https://meganerf.cmusatyalab.org/) models and generate the sparse voxel octrees used by the Mega-NeRF-Dynamic viewer.

The codebase for the Mega-NeRF-Dynamic viewer can be found [here](https://github.com/cmusatyalab/mega-nerf-viewer).

**Note:** This is a preliminary release and there may still be outstanding bugs.

## Citation

```
@misc{turki2021meganerf,
      title={Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs}, 
      author={Haithem Turki and Deva Ramanan and Mahadev Satyanarayanan},
      year={2021},
      eprint={2112.10703},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Demo
![](demo/rubble-orbit.gif)
![](demo/building-orbit.gif)

## Setup

```
conda env create -f environment.yml
conda activate mega-nerf
```

The codebase has been mainly tested against CUDA >= 11.1 and V100/2080 Ti/3090 Ti GPUs. 1080 Ti GPUs should work as well although training will be much slower.

## Data

### Mill 19

- The Building scene can be downloaded [here](https://storage.cmusatyalab.org/mega-nerf-data/building.tgz).
- The Rubble scene can be downloaded [here](https://storage.cmusatyalab.org/mega-nerf-data/rubble.tgz).

### UrbanScene 3D

1. Download the raw photo collections from the [UrbanScene3D](https://vcc.tech/UrbanScene3D/) dataset
2. Download the refined camera poses for one of the scenes below:
  - [Residence](https://storage.cmusatyalab.org/mega-nerf-data/residence.tgz)
  - [Sci-Art](https://storage.cmusatyalab.org/mega-nerf-data/sci-art.tgz)
  - [Campus](https://storage.cmusatyalab.org/mega-nerf-data/campus.tgz)
4. Run ```python scripts/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH```

### Quad 6k Dataset

1. Download the raw photo collections from [here](http://vision.soic.indiana.edu/disco_files/ArtsQuad_dataset.tar).
2. Download [the refined camera poses](https://storage.cmusatyalab.org/mega-nerf-data/quad.tgz)
3. Run ```python scripts/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH```

### Custom Data

We strongly recommend using [PixSFM](https://github.com/cvg/pixel-perfect-sfm) to refine camera poses for your own datasets. Mega-NeRF also assumes that the dataset is properly geo-referenced/aligned such that the second value of its `ray_altitude_range` parameter properly corresponds to ground level. 
If using PixSFM/COLMAP the [model_aligner](https://colmap.github.io/faq.html#geo-registration) utility might be helpful, with [Manhattan world alignment](https://colmap.github.io/faq.html#manhattan-world-alignment) being a possible fallback option if GPS alignment is not possible. 
We provide a [script](https://github.com/cmusatyalab/mega-nerf/blob/main/scripts/colmap_to_mega_nerf.py) to convert from PixSFM/COLMAP output to the format Mega-NeRF expects.

If creating a custom dataset manually, the expected directory structure is:
- /coordinates.pt: [Torch file](https://pytorch.org/docs/stable/generated/torch.save.html) that should contain the following keys:
  - 'origin_drb': Origin of scene in real-world units
  - 'pose_scale_factor': Scale factor mapping from real-world unit (ie: meters) to [-1, 1] range
- '/{val|train}/rgbs/': JPEG or PNG images
- '/{val|train}/metadata/': Image-specific image metadata saved as a torch file. Each image should have a corresponding metadata file with the following file format: {rgb_stem}.pt. Each metadata file should contain the following keys:
  - 'W': Image width
  - 'H': Image height
  - 'intrinsics': Image intrinsics in the following form: [fx, fy, cx, cy]
  - 'c2w': Camera pose. 3x3 camera matrix with the convention used in the original [NeRF repo](https://github.com/bmild/nerf), ie: x: down, y: right, z: backwards, followed by the following transformation: ```torch.cat([camera_in_drb[:, 1:2], -camera_in_drb[:, :1], camera_in_drb[:, 2:4]], -1)```

## Training

1. Generate the training partitions for each submodule: ```python scripts/create_cluster_masks.py --config configs/mega-nerf/${DATASET_NAME}.yml --dataset_path $DATASET_PATH  --output $MASK_PATH --grid_dim $GRID_X $GRID_Y```
    - **Note:** this can be run across multiple GPUs by instead running ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS --max_restarts 0 scripts/create_cluster_masks.py <args>```

예시
`python scripts/create_cluster_masks.py --config configs/mega-nerf/building.yaml --dataset_path data/mill19/building-pixsfm --output output/mill19/building-pixsfm --grid_dim 2 4`


2. Train each submodule: ```python mega_nerf/train.py --config_file configs/mega-nerf/${DATASET_NAME}.yml --exp_name $EXP_PATH --dataset_path $DATASET_PATH --chunk_paths $SCRATCH_PATH --cluster_mask_path ${MASK_PATH}/${SUBMODULE_INDEX}```
    - **Note:** training with against full scale data will write hundreds of GBs / several TBs of shuffled data to disk. You can downsample the training data using ```train_scale_factor``` option.
    - **Note:** we provide [a utility script](parscripts/run_8.txt) based on [parscript](https://github.com/mtli/parscript) to start multiple training jobs in parallel. It can run through the following command: ```CONFIG_FILE=configs/mega-nerf/${DATASET_NAME}.yaml EXP_PREFIX=$EXP_PATH DATASET_PATH=$DATASET_PATH CHUNK_PREFIX=$SCRATCH_PATH MASK_PATH=$MASK_PATH python -m parscript.dispatcher parscripts/run_8.txt -g $NUM_GPUS```
    
    chunck path : 처음에 존재하지 않아야함. 그니까 지정한 폴더 만들어놓지 않고 그냥 만들어질것으로 예상하고 경로 지정하라는 의미 같음. the directory to chunks_path should not initially exist

`python mega_nerf/train.py --config configs/mega-nerf/building.yaml  --exp_name output/mill19/building-pixsfm --dataset_path data/mill19/building-pixsfm --chunk_paths output/mill19/building-pixsfm-chunk --cluster_mask_path output/mill19/building-pixsfm/0`
예를 들어 이렇게 구성한다면 실제 `output/mill19/building-pixsfm-chunk` 폴더는 만들어 놓지 않고 실행
    
    
3. Merge the trained submodules into a unified Mega-NeRF model: ```python scripts/merge_submodules.py --config_file configs/mega-nerf/${DATASET_NAME}.yaml  --ckpt_prefix ${EXP_PREFIX}- --centroid_path ${MASK_PATH}/params.pt --output $MERGED_OUTPUT```

## Evaluation

Single-GPU evaluation: ```python mega_nerf/eval.py --config_file configs/nerf/${DATASET_NAME}.yaml  --exp_name $EXP_NAME --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT```

Multi-GPU evaluation: ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS mega_nerf/eval.py --config_file configs/nerf/${DATASET_NAME}.yaml  --exp_name $EXP_NAME --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT```

## Octree Extraction (for use by Mega-NeRF-Dynamic viewer)

```
python scripts/create_octree.py --config configs/mega-nerf/${DATASET_NAME}.yaml --dataset_path $DATASET_PATH --container_path $MERGED_OUTPUT --output $OCTREE_PATH
 ```

## Acknowledgements

Large parts of this codebase are based on existing work in the [nerf_pl](https://github.com/kwea123/nerf_pl), [NeRF++](https://github.com/Kai-46/nerfplusplus), and [Plenoctree](https://github.com/sxyu/plenoctree) repositories. We use [svox](https://github.com/sxyu/svox) to serialize our sparse voxel octrees and the generated structures should be largely compatible with that codebase.
