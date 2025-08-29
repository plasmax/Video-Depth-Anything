# Video-Depth-Anything for Metric Depth Estimation
We here provide a simple demo for our fine-tuned Video-Depth-Anything metric model. We fine-tune our pre-trained model on Virtual KITTI and IRS datasets for metric depth estimation. 

# Pre-trained Models
We provide our large model:

| Base Model | Params | Checkpoint |
|:-|-:|:-:|
| Metric-Video-Depth-Anything-V2-Large | 381.8M | [Download](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth) |
| Metric-Video-Depth-Anything-V2-base | 113.1M | [Download](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/blob/main/metric_video_depth_anything_vitb.pth) |
| Metric-Video-Depth-Anything-V2-Small | 28.4M | [Download](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/blob/main/metric_video_depth_anything_vits.pth) |

# Metric depth evaluation
We evaluate our model on KITTI and NYU datasets for video metric depth. The evaluation results are as follows.

| δ1 | MogeV2-L | UnidepthV2-L | DepthPro | VDA-S-Metric | VDA-B-Metric | VDA-L-Metric |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| KITTI | 0.415 | **0.982** | 0.822 | 0.877 | 0.887 | *0.910* |
| NYU_v2 | *0.967* | **0.989** | 0.953 | 0.850| 0.883 | 0.908 |

| tae | MogeV2-L | UnidepthV2-L | DepthPro | VDA-S-Metric | VDA-B-Metric | VDA-L-Metric |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| Scannet | 2.56 | 1.41 | 2.73 | 1.48 | *1.26* | **1.09** |


# Usage
## Preparation
```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything.git
cd Video-Depth-Anything
pip3 install -r requirements.txt
cd metric_depth
```
Download the checkpoints and put them under the `checkpoints` directory.

## Use our models
### Running script on video
```bash
python3 run.py \
    --input_video <YOUR_VIDEO_PATH> \
    --output_dir <YOUR_OUTPUT_DIR>
```
### Project video to point clouds
```bash
python3 depth_to_pointcloud.py \
    --input_video <YOUR_VIDEO_PATH> \
    --output_dir <YOUR_OUTPUT_DIR> \
    --focal-length-x <CAMERA FX> \
    --focal-length-y <CAMERA FY> \
```
