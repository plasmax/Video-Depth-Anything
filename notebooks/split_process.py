#!/usr/bin/python3.10

"""
This file separates out Fathom's forward method into individual steps in order to run in parallel on the cpu farm.
Image preprocessing (resizing and normalising) is handled in a separate script as it doesn't involve the model.

Farm stages - each requires the previous one be fully completed before it begins:
1) write out .pt feature files to features temp folder (one task per frame)
2) write out .pt 32-frame blocks to blocks temp folder (one task per FRAME_STEP)
3) write out final depth exrs

"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from fathom.dinov2 import DINOv2
from fathom.dpt_temporal import DPTHeadTemporal

from utils.util import compute_scale_and_shift, get_interpolate_frames
from utils.fast_exr_reader import read_exr_sequence


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


class Fathom(nn.Module):
    def __init__(
        self,
        encoder='vits',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        metric=False,
    ):
        super(Fathom, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.metric = metric

    def export_features(self, output_dir: str, frame: int) -> None:
        """Write out features as .pt files to the specified by out_path.

        output_dir: `str`, e.g. /job/mlearn/common/tmp/fathom/mlast/tmp6w62pm5g
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

        print(f"Loading frame {input_path}")
        
        frame_tensor: torch.Tensor = torch.load(input_path, map_location="cpu", weights_only=True)

        pretrained = DINOv2(model_name=encoder)
        CHECKPOINTS_LOCATION = os.environ["CHECKPOINTS_LOCATION"]
        state_dict = torch.load(f"{CHECKPOINTS_LOCATION}/{checkpoint_name}_{encoder}.pth", map_location="cpu", weights_only=True, mmap=True)
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

        print(f"Features exported for frame {args.frame}.")
    
    def export_temporal_block(self, output_dir: str, frame_step: int, first_frame: int, last_frame: int) -> None:

        # load individual feature .pt files from features_dir and torch.cat them together into the format the DPTHeadTemporal expects
        frame_ids: List[int] = calc_block_feature_frames(frame_step, first_frame, last_frame)  # length 32
        features_dir = Path(output_dir) / "features"
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
        CHECKPOINTS_LOCATION = os.environ["CHECKPOINTS_LOCATION"]
        state_dict = torch.load(f"{CHECKPOINTS_LOCATION}/{checkpoint_name}_{encoder}.pth", map_location="cpu", weights_only=True, mmap=True)
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
            depth = depth.squeeze(1).unflatten(0, (B, T)) # [B, T, H, W]

        blocks_dir = os.path.join(output_dir, "blocks")
        os.makedirs(blocks_dir, exist_ok=True)
        out_path = Path(blocks_dir) / f"block.{frame_step:04d}.pt"
        torch.save(depth, out_path)

    def export_final_exrs(self, input_path: str, publish_path: str, temp_dir: str,  metric: bool = False) -> None:
        """Load all temporal blocks, perform alignment and interpolation, and write final depth EXRs.
        
        Memory-optimized: keeps depth maps at native resolution until final write.
        """

        # 1. Get original frame information
        file_paths = glob.glob(input_path.replace('%04d', '*').replace('#', '*'))
        if not file_paths:
            raise ValueError(f"No input files found matching pattern: {input_path}")

        frame_numbers = sorted([int(f.split(".")[-2]) for f in file_paths])
        first_frame = frame_numbers[0]
        last_frame = frame_numbers[-1]
        org_video_len = len(frame_numbers)

        # Get original resolution from first frame
        first_frame_data = read_exr_sequence(input_path, num_frames=1, max_workers=1)
        _, frame_height, frame_width, _ = first_frame_data.shape

        # 2. Load all blocks from blocks_dir
        blocks_dir = Path(temp_dir) / "blocks"
        if not blocks_dir.exists():
            raise ValueError(f"Blocks directory not found: {blocks_dir}")

        block_files = sorted(blocks_dir.glob("block.*.pt"))
        if not block_files:
            raise ValueError(f"No block files found in {blocks_dir}")

        # Load blocks and extract frame numbers from filenames
        blocks_data = []
        for block_file in block_files:
            frame_num = int(block_file.stem.split(".")[-1])
            block_tensor = torch.load(block_file, map_location="cpu", weights_only=True)
            blocks_data.append((frame_num, block_tensor))

        blocks_data.sort(key=lambda x: x[0])
        block_count = len(blocks_data)

        # FQ_PROGRESS
        total = block_count * 2 + org_video_len
        done = 0

        # 3. Convert blocks to flat depth_list - KEEP AT NATIVE RESOLUTION
        depth_list = []
        for block_idx, (frame_step, block_tensor) in enumerate(blocks_data):
            block_tensor = block_tensor.squeeze(0)  # [T, H, W]
            
            # Store each frame as a tensor at native resolution
            for i in range(block_tensor.shape[0]):
                depth_list.append(block_tensor[i].clone())  # [H, W] tensor

            done += 1
            print(f"FQ_PROGRESS {int(done / total * 100)}%", flush=True)

        # 4. Apply alignment logic at native resolution
        depth_list_aligned = []
        ref_align = []
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        for block_idx in range(block_count):
            frame_id = block_idx * INFER_LEN

            if len(depth_list_aligned) == 0:
                # First block: just add all frames
                depth_list_aligned += [d.clone() for d in depth_list[:INFER_LEN]]
                for kf_id in kf_align_list:
                    ref_align.append(depth_list[frame_id + kf_id].numpy())
            else:
                # Subsequent blocks: apply scale/shift alignment
                curr_align = []
                for i in range(len(kf_align_list)):
                    curr_align.append(depth_list[frame_id + i].numpy())

                if metric:
                    scale, shift = 1.0, 0.0
                else:
                    scale, shift = compute_scale_and_shift(
                        np.concatenate(curr_align),
                        np.concatenate(ref_align),
                        np.concatenate(np.ones_like(ref_align) == 1)
                    )

                # Interpolate overlap region (convert to numpy for interpolation)
                pre_depth_list = [d.numpy() for d in depth_list_aligned[-INTERP_LEN:]]
                post_depth_list = [depth_list[frame_id + align_len + i].numpy() 
                                for i in range(INTERP_LEN)]
                
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i] < 0] = 0
                
                interpolated = get_interpolate_frames(pre_depth_list, post_depth_list)
                # Convert back to tensors
                depth_list_aligned[-INTERP_LEN:] = [torch.from_numpy(d) for d in interpolated]

                # Add remaining frames with scale/shift
                for i in range(OVERLAP, INFER_LEN):
                    new_depth = depth_list[frame_id + i].numpy() * scale + shift
                    new_depth[new_depth < 0] = 0
                    depth_list_aligned.append(torch.from_numpy(new_depth))

                # Update reference alignment
                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    new_depth = depth_list[frame_id + kf_id].numpy() * scale + shift
                    new_depth[new_depth < 0] = 0
                    ref_align.append(new_depth)

            done += 1
            print(f"FQ_PROGRESS {int(done / total * 100)}%", flush=True)

        # Trim to original video length
        depth_list_aligned = depth_list_aligned[:org_video_len]
        
        # Free the unaligned list
        del depth_list

        print(f"Writing depth EXR files to {publish_path}")
        for i, (frame_num, depth_tensor) in enumerate(zip(frame_numbers, depth_list_aligned)):
            # Resize to original resolution at write time
            depth_resized = F.interpolate(
                depth_tensor.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
                size=(frame_height, frame_width),
                mode='bilinear',
                align_corners=True
            ).squeeze().numpy().astype(np.float32)

            # output_path = output_exr_dir / f"depth.{frame_num:04d}.exr"
            output_path = publish_path % frame_num
            cv2.imwrite(str(output_path), depth_resized)

            # Clear reference to allow GC
            depth_list_aligned[i] = None

            if (i + 1) % 10 == 0:
                print(f"  Wrote {i + 1}/{org_video_len} frames")

            done += 1
            print(f"FQ_PROGRESS {int(done / total * 100)}%", flush=True)

        print(f"Successfully exported all {org_video_len} exr files.")


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
    parser = argparse.ArgumentParser(description="Fathom Depth Generator")
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--publish_path', type=str)
    parser.add_argument('--temp_dir', type=str)
    parser.add_argument('--frame', type=int)
    parser.add_argument('--start_frame', type=int)
    parser.add_argument('--end_frame', type=int)
    parser.add_argument('--metric', action='store_true', help='use metric model')

    # stages
    parser.add_argument('--export_features', action='store_true', help='')
    parser.add_argument('--export_blocks', action='store_true', help='')
    parser.add_argument('--export_exrs', action='store_true', help='')

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference running on device: {DEVICE}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    encoder = "vits"
    checkpoint_name = 'metric_fathom' if args.metric else 'fathom'

    fathom = Fathom(**model_configs[encoder], metric=args.metric)

    if args.export_features:
        # Stage 1 - read rgb values, write out .pt feature files to features temp folder (one task per frame)
        fathom.export_features(args.temp_dir, args.frame)

    elif args.export_blocks:
        # Stage 2 - read .pt feature files, write out .pt 32-frame blocks to blocks temp folder
        fathom.export_temporal_block(args.temp_dir, args.frame, args.start_frame, args.end_frame)

    elif args.export_exrs:
        # Stage 3) write out final depth exrs
        fathom.export_final_exrs(args.input_path, args.publish_path, args.temp_dir, metric=args.metric)


