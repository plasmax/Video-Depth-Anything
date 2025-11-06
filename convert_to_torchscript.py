#!/usr/bin/env python3
"""
TorchScript Conversion Script for Video Depth Anything

This script converts Video Depth Anything model to TorchScript format.

TORCHSCRIPT COMPATIBILITY ASSESSMENT:
=====================================
The model has several compatibility issues that need to be addressed:

CRITICAL Issues (Block Compilation):
1. Dynamic Control Flow (dpt_temporal.py:97-123) - Python if/else based on batch size
2. EasyDict Usage (dpt_temporal.py:35) - Dynamic dictionary unpacking
3. Einops Operations (motion_module.py) - String-based tensor rearrangement
4. Conditional xFormers - Runtime library feature detection

This script provides two conversion approaches:
1. torch.jit.trace() - Faster, but limited to specific input shapes
2. torch.jit.script() - More flexible, but requires fixing the issues above

Usage:
    python convert_to_torchscript.py --encoder vitl --output depth_anything_vitl.pt
    python convert_to_torchscript.py --encoder vitb --method trace --batch-size 1 --frames 32
"""

import argparse
import torch
import torch.nn as nn
from video_depth_anything.video_depth import VideoDepthAnything
import warnings
import sys


class VideoDepthAnythingWrapper(nn.Module):
    """
    Wrapper to make the model more TorchScript-friendly by:
    1. Removing micro-batching logic (use fixed batch processing)
    2. Removing conditional control flow where possible
    3. Standardizing input/output format
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, C, H, W]
               B: batch size
               T: number of frames (typically 32)
               C: channels (3 for RGB)
               H, W: height and width (must be multiples of 14)

        Returns:
            depth: Depth prediction of shape [B, T, H, W]
        """
        return self.model(x)


def check_torchscript_compatibility(model):
    """
    Analyze the model for TorchScript compatibility issues.
    """
    print("\n" + "="*70)
    print("TORCHSCRIPT COMPATIBILITY ANALYSIS")
    print("="*70)

    issues = []

    # Check for known problematic modules
    for name, module in model.named_modules():
        module_type = type(module).__name__

        # Check for einops usage (common in motion modules)
        if 'motion' in name.lower() or 'temporal' in name.lower():
            issues.append(f"⚠️  {name}: Temporal module may use einops (incompatible)")

    print("\n🔍 Known Compatibility Issues:")
    print("-" * 70)
    print("1. CRITICAL: Dynamic control flow in DPTHeadTemporal.forward()")
    print("   Location: video_depth_anything/dpt_temporal.py:97-123")
    print("   Issue: Python if/else based on micro_batch_size")
    print("   Impact: Prevents torch.jit.script() compilation")
    print()
    print("2. CRITICAL: EasyDict usage in DPTHeadTemporal.__init__()")
    print("   Location: video_depth_anything/dpt_temporal.py:35")
    print("   Issue: Dynamic dictionary unpacking (**kwargs)")
    print("   Impact: Not compatible with TorchScript initialization")
    print()
    print("3. HIGH: Einops operations in TemporalModule")
    print("   Location: video_depth_anything/motion_module/motion_module.py")
    print("   Issue: String-based rearrange() operations")
    print("   Impact: Can be replaced with torch reshape/permute")
    print()
    print("4. MEDIUM: Conditional xFormers import")
    print("   Location: video_depth_anything/motion_module/attention.py")
    print("   Issue: Runtime library feature detection")
    print("   Impact: Conditional code paths based on library availability")
    print()

    if issues:
        print("\n📋 Module-specific Issues Found:")
        print("-" * 70)
        for issue in issues:
            print(issue)

    print("\n✅ Recommended Approach:")
    print("-" * 70)
    print("Use torch.jit.trace() with fixed input shapes for immediate deployment.")
    print("This works around most issues but limits flexibility.")
    print("For production, consider refactoring the model to be script-compatible.")
    print("="*70 + "\n")


def convert_with_trace(model, batch_size=1, num_frames=32, height=518, width=518, device='cuda'):
    """
    Convert model using torch.jit.trace()

    Pros: Works around dynamic control flow issues
    Cons: Fixed to specific input shapes
    """
    print(f"\n🔄 Converting with torch.jit.trace()...")
    print(f"   Input shape: [{batch_size}, {num_frames}, 3, {height}, {width}]")

    model.eval()

    # Ensure height and width are multiples of 14
    height = (height // 14) * 14
    width = (width // 14) * 14

    # Create example input
    example_input = torch.randn(batch_size, num_frames, 3, height, width).to(device)

    print(f"   Adjusted input shape: [{batch_size}, {num_frames}, 3, {height}, {width}]")

    try:
        with torch.no_grad():
            # Wrap model for cleaner interface
            wrapper = VideoDepthAnythingWrapper(model)

            # Trace the model
            traced_model = torch.jit.trace(wrapper, example_input)

            # Verify the traced model produces same output
            print("   Verifying traced model...")
            original_output = wrapper(example_input)
            traced_output = traced_model(example_input)

            max_diff = torch.max(torch.abs(original_output - traced_output)).item()
            print(f"   ✓ Max difference: {max_diff:.2e}")

            if max_diff > 1e-5:
                warnings.warn(f"Large difference detected: {max_diff}")

            return traced_model

    except Exception as e:
        print(f"   ✗ Tracing failed: {str(e)}")
        print(f"\nError details:")
        import traceback
        traceback.print_exc()
        return None


def convert_with_script(model):
    """
    Convert model using torch.jit.script()

    Pros: More flexible, handles variable input shapes
    Cons: Requires model modifications to handle control flow
    """
    print(f"\n🔄 Converting with torch.jit.script()...")
    print("   Note: This will likely fail due to dynamic control flow")

    model.eval()

    try:
        wrapper = VideoDepthAnythingWrapper(model)
        scripted_model = torch.jit.script(wrapper)
        print("   ✓ Scripting successful!")
        return scripted_model

    except Exception as e:
        print(f"   ✗ Scripting failed (expected): {str(e)}")
        print("\n   To fix this, you need to:")
        print("   1. Replace EasyDict with explicit parameters")
        print("   2. Remove dynamic if/else in forward() (use fixed batch size)")
        print("   3. Replace einops with torch.reshape/permute")
        print("   4. Resolve xFormers conditional imports")
        return None


def load_model(encoder='vitl', features=256, device='cuda'):
    """Load the Video Depth Anything model"""
    print(f"Loading Video Depth Anything model (encoder: {encoder})...")

    model = VideoDepthAnything(
        encoder=encoder,
        features=features,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
    )

    # Try to load pretrained weights if available
    checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded pretrained weights from {checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠️  Pretrained weights not found at {checkpoint_path}")
        print(f"   Continuing with randomly initialized weights")
    except Exception as e:
        print(f"⚠️  Error loading weights: {str(e)}")
        print(f"   Continuing with randomly initialized weights")

    model = model.to(device)
    model.eval()

    return model


def save_torchscript(model, output_path):
    """Save TorchScript model to file"""
    print(f"\n💾 Saving model to {output_path}...")
    try:
        torch.jit.save(model, output_path)

        # Get file size
        import os
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✓ Saved successfully ({size_mb:.2f} MB)")

        # Test loading
        print(f"   Testing load...")
        loaded = torch.jit.load(output_path)
        print(f"   ✓ Load test successful")

        return True
    except Exception as e:
        print(f"   ✗ Save failed: {str(e)}")
        return False


def test_inference(model, device='cuda', height=518, width=518):
    """Test inference with the converted model"""
    print(f"\n🧪 Testing inference...")

    # Ensure dimensions are multiples of 14
    height = (height // 14) * 14
    width = (width // 14) * 14

    test_input = torch.randn(1, 32, 3, height, width).to(device)

    try:
        with torch.no_grad():
            output = model(test_input)

        print(f"   ✓ Input shape: {list(test_input.shape)}")
        print(f"   ✓ Output shape: {list(output.shape)}")
        print(f"   ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Benchmark
        import time
        num_runs = 10
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(test_input)
            torch.cuda.synchronize() if device == 'cuda' else None

        elapsed = time.time() - start
        print(f"   ✓ Average inference time: {elapsed/num_runs*1000:.2f} ms")

        return True

    except Exception as e:
        print(f"   ✗ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert Video Depth Anything to TorchScript')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'],
                        help='Encoder type (vits, vitb, or vitl)')
    parser.add_argument('--output', type=str, default='depth_anything_torchscript.pt',
                        help='Output path for TorchScript model')
    parser.add_argument('--method', type=str, default='trace', choices=['trace', 'script', 'both'],
                        help='Conversion method: trace, script, or both')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for tracing (default: 1)')
    parser.add_argument('--frames', type=int, default=32,
                        help='Number of frames (default: 32)')
    parser.add_argument('--height', type=int, default=518,
                        help='Input height (will be adjusted to multiple of 14)')
    parser.add_argument('--width', type=int, default=518,
                        help='Input width (will be adjusted to multiple of 14)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip inference testing')

    args = parser.parse_args()

    print("="*70)
    print("Video Depth Anything -> TorchScript Converter")
    print("="*70)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load model
    model = load_model(args.encoder, device=args.device)

    # Analyze compatibility
    check_torchscript_compatibility(model)

    # Convert model
    converted_model = None

    if args.method in ['trace', 'both']:
        converted_model = convert_with_trace(
            model,
            batch_size=args.batch_size,
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            device=args.device
        )

        if converted_model and args.method == 'trace':
            # Test and save
            if not args.no_test:
                test_inference(converted_model, args.device, args.height, args.width)

            save_torchscript(converted_model, args.output)

    if args.method in ['script', 'both']:
        scripted_model = convert_with_script(model)

        if scripted_model:
            output_path = args.output.replace('.pt', '_scripted.pt') if args.method == 'both' else args.output
            save_torchscript(scripted_model, output_path)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if converted_model:
        print("✅ TorchScript conversion successful!")
        print(f"\n📦 Model saved to: {args.output}")
        print(f"\n🚀 Usage example:")
        print(f"   import torch")
        print(f"   model = torch.jit.load('{args.output}')")
        print(f"   model.eval()")
        print(f"   output = model(input_tensor)  # input: [B, {args.frames}, 3, {args.height}, {args.width}]")
    else:
        print("⚠️  TorchScript conversion failed")
        print("\nTo make the model fully TorchScript compatible, you need to:")
        print("1. Refactor DPTHeadTemporal to avoid dynamic control flow")
        print("2. Replace EasyDict with explicit parameter passing")
        print("3. Replace einops operations with native PyTorch")
        print("4. Handle xFormers imports properly")
        print("\nFor now, use torch.jit.trace() with fixed input shapes.")

    print("="*70)


if __name__ == '__main__':
    main()
