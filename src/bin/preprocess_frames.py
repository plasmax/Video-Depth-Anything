import os
import glob
import argparse
import numpy as np
import OpenEXR
import Imath
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from functools import partial


def preprocess_frames(
    input_path: str,
    output_dir: Optional[str] = None,
    input_size: int = 518,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> None:
    """Export EXRs at the resolution they will be processed by the model.

    Memory-efficient: only keeps as many frames in memory as there are CPU threads.

    Args:
        input_path: Path pattern with %04d or # for frame number
        output_dir: Directory to write .exr files
        input_size: Target size for model input
        start_frame: First frame to process (inclusive). None = first available.
        end_frame: Last frame to process (inclusive). None = last available.
    """
    # Resolve file paths and frame numbers
    search_pattern = input_path.replace('%04d', '*').replace('#', '*')
    file_paths = sorted(glob.glob(search_pattern))
    
    if not file_paths:
        raise FileNotFoundError(f"No files found for pattern: {search_pattern}")
    
    frame_numbers = [int(f.split(".")[-2]) for f in file_paths]
    
    # Filter by frame range
    if start_frame is not None or end_frame is not None:
        filtered = [
            (p, n) for p, n in zip(file_paths, frame_numbers)
            if (start_frame is None or n >= start_frame) and (end_frame is None or n <= end_frame)
        ]
        if not filtered:
            raise ValueError(f"No frames found in range [{start_frame}, {end_frame}]")
        file_paths, frame_numbers = zip(*filtered)
        file_paths = list(file_paths)
        frame_numbers = list(frame_numbers)
    
    # Determine worker count
    max_workers = os.getenv("FQ_CPUS", None)
    max_workers = int(max_workers) if max_workers else os.cpu_count()
    
    # Read first frame to get dimensions for transform params
    first_frame = _read_exr_frame(file_paths[0])
    frame_height, frame_width = first_frame.shape[:2]
    del first_frame  # Free immediately
    
    # Adjust input_size for extreme aspect ratios
    ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
    if ratio > 1.78:
        input_size = int(input_size * 1.777 / ratio)
        input_size = round(input_size / 14) * 14
    
    # Precompute transform parameters (avoid pickling transform objects)
    transform_params = _compute_transform_params(frame_height, frame_width, input_size)
    
    # Setup output directory
    features_dir = None
    if output_dir is not None:
        features_dir = os.path.join(output_dir, "rgb")
        os.makedirs(features_dir, exist_ok=True)
    
    # Create worker function with baked-in params
    worker_fn = partial(
        _process_and_transform_frame,
        transform_params=transform_params,
        output_dir=features_dir
    )
    
    # Process frames with bounded concurrency
    # ProcessPoolExecutor naturally limits memory: only max_workers tasks run at once,
    # and we submit new work only as old work completes
    frame_range = f"[{frame_numbers[0]}-{frame_numbers[-1]}]" if len(frame_numbers) > 1 else f"[{frame_numbers[0]}]"
    print(f"Processing {len(file_paths)} frames {frame_range} with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        total = len(file_paths)
        for i, result in enumerate(executor.map(worker_fn, file_paths, frame_numbers)):
            done = i + 1
            print(f"FQ_PROGRESS {int(done / total * 100)}%", flush=True)


def _compute_transform_params(height: int, width: int, input_size: int) -> dict:
    """Precompute resize dimensions to avoid pickling transform objects."""
    # Replicate Resize logic from depth_anything
    scale = input_size / min(height, width)
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Ensure multiple of 14
    new_height = (new_height // 14) * 14
    new_width = (new_width // 14) * 14
    
    # Ensure at least input_size (lower_bound behavior)
    if new_height < input_size:
        new_height = ((input_size + 13) // 14) * 14
    if new_width < input_size:
        new_width = ((input_size + 13) // 14) * 14
    
    return {
        'new_height': new_height,
        'new_width': new_width,
        'mean': np.array([0.485, 0.456, 0.406], dtype=np.float32),
        'std': np.array([0.229, 0.224, 0.225], dtype=np.float32),
    }


def _read_exr_frame(path: str) -> np.ndarray:
    """Read and convert a single EXR frame to uint8 RGB."""
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()

    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read RGB channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = exr_file.channels(['R', 'G', 'B'], FLOAT)

    # Convert to numpy arrays
    r = np.frombuffer(channels[0], dtype=np.float32).reshape(height, width)
    g = np.frombuffer(channels[1], dtype=np.float32).reshape(height, width)
    b = np.frombuffer(channels[2], dtype=np.float32).reshape(height, width)

    # Stack to HWC format
    frame = np.stack([r, g, b], axis=2)

    # Sanitize floats
    np.nan_to_num(frame, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(frame, 0.0, 1.0, out=frame)

    # sRGB gamma correction (linear to gamma)
    mask = frame <= 0.0031308
    frame_pow = np.power(frame, 1/2.4)
    frame_pow *= 1.055
    frame_pow -= 0.055
    frame_linear = frame * 12.92
    frame = np.where(mask, frame_linear, frame_pow)

    # Convert to uint8
    frame *= 255.0
    np.clip(frame, 0, 255, out=frame)

    return frame.astype(np.uint8)


def _process_and_transform_frame(
    path: str,
    frame_num: int,
    transform_params: dict,
    output_dir: Optional[str]
) -> Optional[str]:
    """Worker: read, transform, and save a single frame. Runs in subprocess."""
    # Read frame
    frame = _read_exr_frame(path)

    # Convert to float and normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0

    # Resize using PIL (BICUBIC interpolation)
    img = Image.fromarray((frame * 255).astype(np.uint8))
    img = img.resize(
        (transform_params['new_width'], transform_params['new_height']),
        Image.BICUBIC
    )
    frame = np.array(img).astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    frame = (frame - transform_params['mean']) / transform_params['std']

    # PrepareForNet: HWC -> CHW
    frame = frame.transpose(2, 0, 1)

    # Save if output directory provided
    if output_dir is not None:
        out_path = os.path.join(output_dir, f"rgb.{frame_num:04d}.exr")
        _write_exr_frame(out_path, frame)
        return out_path

    return None


def _write_exr_frame(path: str, data: np.ndarray) -> None:
    """Write a CHW numpy array to EXR file.

    Args:
        path: Output file path
        data: Array in CHW format (channels, height, width)
    """
    # Ensure float32
    data = data.astype(np.float32)

    # Get dimensions
    channels, height, width = data.shape

    # Create header
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }

    # Create output file
    exr_file = OpenEXR.OutputFile(path, header)

    # Write channels (CHW -> separate channel buffers)
    channel_data = {
        'R': data[0].tobytes(),
        'G': data[1].tobytes(),
        'B': data[2].tobytes()
    }

    exr_file.writePixels(channel_data)
    exr_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--temp_dir', type=str)
    parser.add_argument('--start_frame', type=int)
    parser.add_argument('--end_frame', type=int)

    args = parser.parse_args()

    preprocess_frames(args.input_path, 
                      output_dir=args.temp_dir, 
                      start_frame=args.start_frame, 
                      end_frame=args.end_frame)
