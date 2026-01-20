# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import numpy as np
import os
import torch

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')
    
    # LoRA Fine-Tuning Arguments
    parser.add_argument('--fine_tune', action='store_true', help='Fine-tune the model on the input video sequence using LoRA.')
    parser.add_argument('--lora_rank', type=int, default=4, help='Rank of LoRA adapters.')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of fine-tuning epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for fine-tuning.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for spatial consistency loss.')
    parser.add_argument('--stable_scale', type=float, default=10.0, help='Weight for temporal stability loss.')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    
    frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    
    if args.fine_tune:
        print(f"Starting LoRA fine-tuning (Rank={args.lora_rank}, Epochs={args.train_epochs}) to optimize for this specific sequence...")
        
        # 1. Generate Pseudo-GT (Self-Distillation)
        print("Generating Pseudo-GT from base model...")
        with torch.no_grad():
            # We run inference to get the base predictions
            # Note: infer_video_depth returns numpy array, we need it as tensor for training if possible,
            # but for simplicity, we can just use the output as static targets.
            # Ideally, we should run the forward pass in the loop, but memory might be an issue.
            # Let's use the infer_video_depth to get the full consistent sequence first.
            pseudo_depths, _ = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
            
        # Convert frames and pseudo_depths to torch tensors for training
        # frames: (T, H, W, 3) -> (T, 3, H, W) normalized
        # depths: (T, H, W) -> (T, 1, H, W)
        
        # We need to replicate the transform logic from video_depth.py
        from torchvision.transforms import Compose
        from video_depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        import cv2
        
        # Reuse size logic
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        input_size = args.input_size
        if ratio > 1.78:
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
        
        # Prepare dataset
        train_images = []
        for i in range(frames.shape[0]):
            img = transform({'image': frames[i].astype(np.float32) / 255.0})['image'] # (3, H, W)
            train_images.append(torch.from_numpy(img))
        train_images = torch.stack(train_images) # (T, 3, H, W)
        
        # Prepare targets (Pseudo-Labels)
        # We need to resize pseudo_depths to the same input_size usually, OR we compute loss at output size.
        # The model outputs at input_size (patch-wise), then interpolates. 
        # Let's compute loss at model output resolution to save memory, or interpolate targets.
        # Simpler: Interpolate targets to train_images size.
        train_targets = torch.from_numpy(pseudo_depths).unsqueeze(1) # (T, 1, H, W)
        # We might need to resize targets to match the model's expected input/output if they differ slightly due to padding?
        # Actually, the model forward returns (B, T, H_in, W_in) usually before resize.
        # Let's check model forward:
        # depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True) where H,W is input x shape.
        
        # So model output matches input tensor shape.
        # train_targets were generated by 'infer_video_depth' which resizes BACK to original resolution.
        # So we should resize train_targets to match train_images.
        H_in, W_in = train_images.shape[-2:]
        train_targets = torch.nn.functional.interpolate(train_targets, size=(H_in, W_in), mode='bilinear', align_corners=True)
        
        # 2. Apply LoRA
        video_depth_anything.apply_lora(rank=args.lora_rank)
        video_depth_anything = video_depth_anything.to(DEVICE).train() # Set to train mode (only LoRA grads enabled)
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, video_depth_anything.parameters()), lr=args.learning_rate)
        
        from loss.loss import VideoDepthLoss
        # We use existing losses. 
        # spatial_loss (Procrustes) helps maintain structure (Consistency with Pseudo-GT).
        # stable_loss (TemporalGradientMatching) helps reduce flickering.
        criterion = VideoDepthLoss(alpha=args.alpha, trim=0.0, stable_scale=args.stable_scale, reduction='batch-based')
        
        # 3. Training Loop
        # We process in batches of INFER_LEN usually, but for fine-tuning we might want smaller batches or full sequence if fits.
        # Run.py default calls infer_video_depth which handles windowing.
        # Here we need a simplified loop. Let's do a simple sliding window or batches.
        
        batch_size = 4 # Small batch for training
        
        print(f"Training for {args.train_epochs} epochs...")
        for epoch in range(args.train_epochs):
            total_loss = 0
            num_batches = 0
            
            # Shuffle or sequential? Sequential is better for temporal loss if we pass valid masks.
            # But the model inputs (B, T, C, H, W). Run.py uses T as sequence length.
            # The model expects T frames.
            # Let's assume we train on chunks of T=8 or T=16.
            
            T_train = 8
            for i in range(0, len(train_images) - T_train + 1, 4): # Stride 4
                batch_imgs = train_images[i:i+T_train].unsqueeze(0).to(DEVICE) # (1, T, 3, H, W)
                batch_targets = train_targets[i:i+T_train].unsqueeze(0).to(DEVICE) # (1, T, 1, H, W)
                
                # Check valid mask (dummy all ones for now)
                mask = torch.ones_like(batch_targets)
                
                optimizer.zero_grad()
                
                # Forward
                # model forward expects (B, T, C, H, W)
                preds = video_depth_anything(batch_imgs) # (1, T, H, W)
                preds = preds.unsqueeze(2) # (1, T, 1, H, W)
                
                # Loss
                loss_dict = criterion(preds, batch_targets, mask)
                loss = loss_dict['total_loss']
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            print(f"Epoch {epoch+1}/{args.train_epochs}, Loss: {total_loss/num_batches:.4f}")
            
        print("Fine-tuning completed. Switching to eval mode.")
        video_depth_anything.eval()
        
        # No need to reload, weights are updated in place. (LoRA weights)
    
    # Run inference with the fine-tuned model
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)

    video_name = os.path.basename(args.input_video)
    os.makedirs(args.output_dir, exist_ok=True)

    processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)

    if args.save_npz:
        depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
        np.savez_compressed(depth_npz_path, depths=depths)
    if args.save_exr:
        depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(depth_exr_dir, exist_ok=True)
        import OpenEXR
        import Imath
        for i, depth in enumerate(depths):
            output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
            header = OpenEXR.Header(depth.shape[1], depth.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth.tobytes()})
            exr_file.close()

    if args.metric:
        import open3d as o3d

        width, height = depths[0].shape[-1], depths[0].shape[-2]
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width / 2) / args.focal_length_x
        y = (y - height / 2) / args.focal_length_y

        for i, (color_image, depth) in enumerate(zip(frames, depths)):
            z = np.array(depth)
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = np.array(color_image).reshape(-1, 3) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(os.path.join(args.output_dir, 'point' + str(i).zfill(4) + '.ply'), pcd)
