#!/job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/Video-Depth-Anything/py10venv_el9/bin/python

"""
This file separates out VideoDepthAnything's forward method into individual steps in order to test parallelization on the cpu farm.

Pre-submission:
  calculate batch indexes
  calculate image resolution that the model will process at
  setup temp directories - rgb, features, blocks

Farm stages - each requires the previous one be fully completed before it begins:
1) write out RGB files at that resolution - use preprocess_frames method
2) write out .pt feature files to features temp folder (one task per frame)
3) write out .pt 32-frame blocks to blocks temp folder (one task per FRAME_STEP)
4) write out final depth exrs
5) cleanup stage to remove temp files

This can also be used locally to take advantage of hardware accelerators.


Example commands for testing:

1)
(cd /job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/Video-Depth-Anything && ./py10venv_el9/bin/python ./notebook/split_process.py --prep_frames --input_path "/job/mlearn/vault/vfx_image_sequence/dev/depthestimation/sdfx/asset_sourceplates/mp01_default_2484x1176_ldn.mlearn.asset.1407801/v002/main/dev_gef056_0670_sourceplates_mp01_v002_main.%04d.exr" --output_dir "/job/mlearn/dev/depthestimation/VideoDepthAnything_Test01/work/mlast/python/temp_out")
2) 
(cd /job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/Video-Depth-Anything && ./py10venv_el9/bin/python ./notebook/split_process.py --export_features --input_path "/job/mlearn/vault/vfx_image_sequence/dev/depthestimation/sdfx/asset_sourceplates/mp01_default_2484x1176_ldn.mlearn.asset.1407801/v002/main/dev_gef056_0670_sourceplates_mp01_v002_main.%04d.exr" --output_dir "/job/mlearn/dev/depthestimation/VideoDepthAnything_Test01/work/mlast/python/temp_out" --frame "<SPACEDFRAMELIST>")
3)
(cd /job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/Video-Depth-Anything && ./py10venv_el9/bin/python ./notebook/split_process.py --export_blocks --output_dir "/job/mlearn/dev/depthestimation/VideoDepthAnything_Test01/work/mlast/python/temp_out" --frame "<SPACEDFRAMELIST>" --start_frame 977 --end_frame 1113)
4)

5)


"""

import sys
import os
import glob

# TODO: remove hard coded path before deploying, video_depth_anything module will be available on PATH.
sys.path.append("/job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/Video-Depth-Anything/")
from typing import Optional, Sequence, Tuple, Union, List
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc

from video_depth_anything.dinov2 import DINOv2
from video_depth_anything.dpt_temporal import DPTHeadTemporal
from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames
from utils.fast_exr_reader import read_exr_sequence, get_first_frame


# Type alias: per-layer (patch_tokens, class_token)
LayerFeat = Tuple[torch.Tensor, torch.Tensor]
# Full feature tuple: one entry per intermediate layer
FrameFeat = Tuple[LayerFeat, ...]

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8
FRAME_STEP = INFER_LEN - OVERLAP


class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        metric=False,
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.metric = metric

    def preprocess_frames(self, input_path: str, output_dir: Optional[str] = None, input_size: int = 518) -> None:
        """Export exrs at the resolution they will be processed by the model.

        input_path: e.g. /job/mlearn/vault/vfx_image_sequence/dev/depthestimation/sdfx/asset_sourceplates/mp01_default_2484x1176_ldn.mlearn.asset.1407801/v002/main/dev_gef056_0670_sourceplates_mp01_v002_main.%04d.exr
        output_dir: e.g. /job/mlearn/dev/depthestimation/VideoDepthAnything_Test01/work/mlast/python/temp_out
        
        """
        path = input_path.replace('%04d', '*').replace('#', '*')
        file_paths = glob.glob(input_path.replace('%04d', '*').replace('#', '*'))
        frame_numbers = sorted([int(f.split(".")[-2]) for f in file_paths])

        cpus = os.getenv("FQ_CPUS", None)
        if cpus is not None:
            cpus = int(cpus)
        frames = read_exr_sequence(input_path, max_workers=cpus)

        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = []
        for i in range(frames.shape[0]):
            frame = torch.from_numpy(transform({'image': frames[i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0)
            frame_list.append(frame)

        # write .pt files to the temp folder
        if output_dir is not None:
            if os.path.exists(output_dir):
                features_dir = os.path.join(output_dir, "rgb")
                os.makedirs(features_dir, exist_ok=True)
                for i, data in zip(frame_numbers, frame_list):
                    out_path = f"{features_dir}/rgb.{i:04d}.pt"
                    print(out_path)
                    torch.save(data, out_path)

        return torch.cat(frame_list, dim=1) # leave this here for now so we can verify tensor sizes for the next step

    def export_features(self, output_dir: str, frame: int) -> None:
        """Write out features as .pt files to the specified by out_path.

        output_dir: `str`, e.g. /job/mlearn/dev/depthestimation/VideoDepthAnything_Test01/work/mlast/python/temp_out/
            output_dir should contain a folder with files like:
                rgb/rgb.1001.pt
            and we are going to write:
                features/features.1001.pt

        This function expects that the model's state_dict have NOT been loaded into the model yet.
        """
        rgb_folder = os.path.join(output_dir, "rgb")
        if not os.path.exists(rgb_folder):
            raise ValueError(f"Expecting rgb images to have been written to folder: \n\n{rgb_folder}")

        input_path = None
        for filename in os.listdir(rgb_folder):
            if frame == int(filename.split(".")[-2]):
                input_path = os.path.join(rgb_folder, filename)
        if input_path is None:
            raise ValueError(f"Couldn't find frame {rgb_folder}/rgb.{i:04d}.pt")
        
        frame_tensor: torch.Tensor = torch.load(input_path, map_location="cpu", weights_only=True)

        pretrained = DINOv2(model_name=encoder)
        state_dict = torch.load(f'./checkpoints/{checkpoint_name}_{encoder}.pth', map_location='cpu', weights_only=True, mmap=True)
        encoder_dict = {
            k.replace("pretrained.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("pretrained.")
        }
        pretrained.load_state_dict(encoder_dict, strict=True)
        pretrained.half()
        pretrained.to(DEVICE).eval()

        frame_tensor = frame_tensor.half()
        frame_tensor = frame_tensor.to(DEVICE)
        intermediate_layer_idx = {
                    'vits': [2, 5, 8, 11],
                    "vitb": [2, 5, 8, 11],
                    'vitl': [4, 11, 17, 23]
                }
        layers = intermediate_layer_idx[encoder]

        # from the VDA paper: we collapse B x T into the batch dimension, to get B,C,H,W
        frame_tensor = frame_tensor.flatten(0, 1)
        with torch.no_grad():
            features = pretrained.get_intermediate_layers(frame_tensor, layers, return_class_token=True)

        features_dir = os.path.join(output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        out_path = f"{features_dir}/features.{frame:04d}.pt"
        torch.save(features, out_path)
    
    def export_temporal_block(self, output_dir: str, frame_step: int, first_frame: int, last_frame: int) -> None:

        # load individual feature .pt files from features_dir and torch.cat them together into the format the DPTHeadTemporal expects
        frame_ids: List[int] = calc_block_feature_frames(frame_step, first_frame, last_frame)  # length 32
        features_dir = Path(output_dir) / "features" #/ "features.%04d.pt"
        features = load_and_stack_temporal_features(frame_ids, features_dir, map_location=DEVICE)

        # get resized image dimensions from previous rgb tensor
        rgb_path = Path(output_dir) / "rgb" / f"rgb.{frame_step:04d}.pt"
        rgb_tensor = torch.load(rgb_path, map_location="cpu", weights_only=True)
        _,_,C,H,W = rgb_tensor.shape
        B,T = 1, 32

        out_channels = model_configs[encoder]["out_channels"]
        use_bn = False
        use_clstoken = False
        num_frames = 32
        pe = 'ape'
        head = DPTHeadTemporal(self.pretrained.embed_dim, 
                               model_configs[encoder]["features"], 
                               use_bn, 
                               out_channels=out_channels, 
                               use_clstoken=use_clstoken, 
                               num_frames=num_frames, 
                               pe=pe)
        state_dict = torch.load(f'./checkpoints/{checkpoint_name}_{encoder}.pth', map_location='cpu', weights_only=True, mmap=True)
        decoder_dict = {
            k.replace("head.", ""): v 
            for k, v in state_dict.items() 
            if k.startswith("head.")
        }
        head.load_state_dict(decoder_dict, strict=True)

        head = head.half()
        head = head.to(DEVICE)
        patch_h, patch_w = H // 14, W // 14

        with torch.no_grad():
            depth = head(features, patch_h, patch_w, T)[0] # [32, 1, H, W]
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
            depth = F.relu(depth) # [32, 1, H, W]
            depth = depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]

        blocks_dir = os.path.join(output_dir, "blocks")
        os.makedirs(blocks_dir, exist_ok=True)
        out_path = Path(blocks_dir) / f"block.{frame_step:04d}.pt"
        torch.save(depth, out_path)

    def export_final_exrs(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """VideoDepthAnything's original forward method. Left here for local/GPU usage."""
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]

    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        """VideoDepthAnything's original infer_video_depth method. Left here for local/GPU usage."""
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:  # we recommend to process video with ratio smaller than 16:9 due to memory limitation
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        frame_list = [frames[i] for i in range(frames.shape[0])]
        org_video_len = len(frame_list)
        append_frame_len = (FRAME_STEP - (org_video_len % FRAME_STEP)) % FRAME_STEP + (INFER_LEN - FRAME_STEP)
        frame_list = frame_list + [frame_list[-1].copy()] * append_frame_len

        depth_list = []
        pre_input = None
        for frame_id in tqdm(range(0, org_video_len, FRAME_STEP)):
            cur_list = []
            for i in range(INFER_LEN):
                cur_frame = torch.from_numpy(transform({'image': frame_list[frame_id+i].astype(np.float32) / 255.0})['image']).unsqueeze(0).unsqueeze(0)
                print(cur_frame.shape, cur_frame.dtype, cur_frame.device)
                cur_list.append(cur_frame)
            cur_input = torch.cat(cur_list, dim=1).to(device)
            print(FRAME_STEP, cur_input.shape)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...] # TODO: if we want to batch this on the farm, this logic should be moved to the exr reading step.

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input) # depth shape: [1, T, H, W]

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0,1).unsqueeze(1), size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            depth_list += [depth[i][0].cpu().numpy() for i in range(depth.shape[0])]

            pre_input = cur_input

        del frame_list
        gc.collect()

        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for frame_id in range(0, len(depth_list), INFER_LEN):
            if len(depth_list_aligned) == 0:
                depth_list_aligned += depth_list[:INFER_LEN]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id+kf_id])
            else:
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id+i])

                if self.metric:
                    scale, shift = 1.0, 0.0
                else:
                    scale, shift = compute_scale_and_shift(np.concatenate(curr_align),
                                                           np.concatenate(ref_align),
                                                           np.concatenate(np.ones_like(ref_align)==1))

                pre_depth_list = depth_list_aligned[-INTERP_LEN:]
                post_depth_list = depth_list[frame_id+align_len:frame_id+OVERLAP]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i]<0] = 0
                depth_list_aligned[-INTERP_LEN:] = get_interpolate_frames(pre_depth_list, post_depth_list)

                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id+i] * scale + shift
                    new_depth[new_depth<0] = 0
                    depth_list_aligned.append(new_depth)

                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id+kf_id] * scale + shift
                    new_depth[new_depth<0] = 0
                    ref_align.append(new_depth)

        depth_list = depth_list_aligned

        return np.stack(depth_list[:org_video_len], axis=0), target_fps



def get_chunk_structure(total_frames):
    # Handle padding logic exactly as in the original code
    append_frame_len = (FRAME_STEP - (total_frames % FRAME_STEP)) % FRAME_STEP + (INFER_LEN - FRAME_STEP)
    padded_len = total_frames + append_frame_len
    
    # Create list of actual video indices (use -1 or similar to denote padded frames if needed, 
    # or just clamp to total_frames-1)
    all_indices = list(range(total_frames)) + [total_frames - 1] * append_frame_len
    
    chunks = []
    prev_chunk_indices = None
    
    # Simulate the sliding window
    for frame_id in range(0, total_frames, FRAME_STEP):
        # 1. Get the "New" frames for this window
        # In the original code, this comes from frame_list[frame_id : frame_id+INFER_LEN]
        # BUT, the first OVERLAP frames are immediately overwritten if prev exists.
        # So we really only care about the tail.
        curr_chunk_indices = all_indices[frame_id : frame_id + INFER_LEN]
        
        # 2. Apply Keyframe injection from previous chunk
        if prev_chunk_indices is not None:
            # Construct the context part
            context_indices = [prev_chunk_indices[k] for k in KEYFRAMES]
            # Overwrite the first OVERLAP frames
            curr_chunk_indices[:OVERLAP] = context_indices
            
        chunks.append(curr_chunk_indices)
        prev_chunk_indices = curr_chunk_indices
        
    return chunks


def _clamp_frame(f: int, first_frame: int, last_frame: int) -> int:
    if f < first_frame:
        return first_frame
    if f > last_frame:
        return last_frame
    return f

def calc_block_feature_frames(
    frame_step: int,
    first_frame: int,
    last_frame: int,
) -> List[int]:
    """
    For a given block start frame_step (absolute), return the 32 absolute frame
    numbers of features.%04d.pt to load to build cur_input for DPTHeadTemporal.

    - Base window: frame_step..frame_step+31 (clamped at end by repeating last_frame)
    - If not the first block: overwrite first OVERLAP entries using the previous block's KEYFRAMES
      (equivalent to: cur_input[:, :OVERLAP] = pre_input[:, KEYFRAMES]).
    """
    if last_frame < first_frame:
        raise ValueError("last_frame must be >= first_frame")

    # Basic 32-frame window for this block
    frames = [_clamp_frame(frame_step + i, first_frame, last_frame) for i in range(INFER_LEN)]

    # If it's not the very first block, apply the overwrite mapping.
    if frame_step != first_frame:
        prev_start = frame_step - FRAME_STEP

        # Optional: sanity check you're on the block grid
        if (frame_step - first_frame) % FRAME_STEP != 0:
            raise ValueError(
                f"frame_step {frame_step} is not on the block grid starting at {first_frame} with step {FRAME_STEP}"
            )

        # This reproduces the exact dependency behaviour:
        # - index 0 comes from the global first frame (propagated through all blocks)
        # - the rest come from prev_start + [12, 24..31]
        overwrite_sources = [first_frame] + [_clamp_frame(prev_start + k, first_frame, last_frame) for k in KEYFRAMES[1:]]
        frames[:OVERLAP] = overwrite_sources  # length 10

    return frames



def load_and_stack_temporal_features(
    frame_ids: Sequence[int],
    features_dir: str | Path,
    map_location: str | torch.device = "cpu",
) -> FrameFeat:
    """
    Load per-frame saved encoder features and stack them into the exact structure
    expected by DPTHeadTemporal(..., T).

    Input:
      - frame_ids: length T (e.g. 32), absolute frame numbers
      - features_dir: directory containing features.%04d.pt
        (saved as the encoder output: tuple((patch, cls), ...))
    Output:
      - features: tuple over layers: (patch_T, cls_T)
        where:
          patch_T: [B*T, N, C]   (B is 1 in your pipeline)
          cls_T:   [B*T, C]
    """
    features_dir = Path(features_dir)

    # Load first frame to establish structure
    first_path = features_dir / f"features.{int(frame_ids[0]):04d}.pt"
    first: FrameFeat = torch.load(first_path, map_location=map_location, weights_only=True)

    num_layers = len(first)
    patches_per_layer: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]
    cls_per_layer: List[List[torch.Tensor]] = [[] for _ in range(num_layers)]

    def _validate_frame_feat(ff: FrameFeat, src: str) -> None:
        if not isinstance(ff, (tuple, list)) or len(ff) != num_layers:
            raise ValueError(f"{src}: expected {num_layers} layers, got {type(ff)} len={len(ff) if hasattr(ff,'__len__') else '??'}")
        for li, (p, c) in enumerate(ff):
            if p.dim() != 3:
                raise ValueError(f"{src}: layer {li} patch tensor must be [B,N,C], got {tuple(p.shape)}")
            if c.dim() != 2:
                raise ValueError(f"{src}: layer {li} cls tensor must be [B,C], got {tuple(c.shape)}")
            if p.shape[0] != 1 or c.shape[0] != 1:
                raise ValueError(f"{src}: expected B=1, got patch B={p.shape[0]} cls B={c.shape[0]}")

    # Include the first one
    _validate_frame_feat(first, str(first_path))
    for li, (p, c) in enumerate(first):
        patches_per_layer[li].append(p)
        cls_per_layer[li].append(c)

    # Load remaining frames
    for fid in frame_ids[1:]:
        pth = features_dir / f"features.{int(fid):04d}.pt"
        ff: FrameFeat = torch.load(pth, map_location=map_location, weights_only=True)
        _validate_frame_feat(ff, str(pth))
        for li, (p, c) in enumerate(ff):
            patches_per_layer[li].append(p)
            cls_per_layer[li].append(c)

    # Stack time -> [T, B, ...] then flatten -> [B*T, ...] to match x.flatten(0,1) convention
    out: List[LayerFeat] = []
    for li in range(num_layers):
        patch_t = torch.cat(patches_per_layer[li], dim=0)  # [T, N, C] because each was [1,N,C]
        cls_t   = torch.cat(cls_per_layer[li], dim=0)      # [T, C]   because each was [1,C]

        # Make it explicitly [B*T, ...] with B=1
        patch_bt = patch_t.contiguous()  # [32, N, C]
        cls_bt   = cls_t.contiguous()    # [32, C]

        out.append((patch_bt, cls_bt))

    return tuple(out)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--output_feature_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--frame', type=int)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--autocrop', action='store_true', help='crop black bars')
    parser.add_argument('--start_frame', type=int)
    parser.add_argument('--end_frame', type=int)

    # stages. these will be broken into individual files
    parser.add_argument('--plan', action='store_true', help='Planning/job prep stage')
    parser.add_argument('--prep_frames', action='store_true', help='Prepare/resize frames')
    parser.add_argument('--export_features', action='store_true', help='')
    parser.add_argument('--export_blocks', action='store_true', help='')
    parser.add_argument('--export_exrs', action='store_true', help='')

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INPUT_PATH = "/job/mlearn/vault/vfx_image_sequence/dev/depthestimation/sdfx/asset_sourceplates/mp01_default_2484x1176_ldn.mlearn.asset.1407801/v002/main/dev_gef056_0670_sourceplates_mp01_v002_main.%04d.exr"
    TEMP_DIR = "/job/mlearn/dev/depthestimation/VideoDepthAnything_Test01/work/mlast/python/temp_out"

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    encoder = "vits"
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    if args.plan:
        # Pre-submission:
        #   calculate batch indexes
        #   calculate image resolution that the model will process at
        #   setup temp directories - rgb, features, blocks

        # Set up farm stages - each requires the previous one be fully completed before it begins:

        chunks = get_chunk_structure(127)
        from pprint import pprint
        pprint(chunks)
        # Chunk 0 needs frames: [0, 1, ..., 31]
        # Chunk 1 needs frames: [0, 12, 24..., 32, 33...]
    else:
        video_depth_anything = VideoDepthAnything(**model_configs[encoder], metric=args.metric)
        if args.prep_frames:
            # 1) write out RGB files at that resolution - use preprocess_frames method
            frames = video_depth_anything.preprocess_frames(args.input_path, args.output_dir)
            print(frames.shape)

        elif args.export_features:
            # 2) write out .pt feature files to features temp folder (one task per frame)
            video_depth_anything.export_features(args.output_dir, args.frame)
            print(f"block exported for frame {args.frame}")

        elif args.export_blocks:
            # 3) write out .pt 32-frame blocks to blocks temp folder (one task, unfortunately needs to be done sequentially)
            video_depth_anything.export_temporal_block(args.output_dir, args.frame, args.start_frame, args.end_frame)

        elif args.export_exrs:
            # 4) write out final depth exrs
            video_depth_anything.export_final_exrs



