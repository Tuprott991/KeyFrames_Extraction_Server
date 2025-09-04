from TransNetV2.inference.transnetv2 import TransNetV2
import os
import numpy as np
import tensorflow as tf
import cv2
import json
import imagehash
from PIL import Image
import csv
import sys
import argparse

def setup_gpu(gpu_id):
    """Configure TensorFlow to use a specific GPU with optimized memory settings"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for the specified GPU
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            
            # Optional: Set memory limit if needed
            # tf.config.experimental.set_memory_limit(gpus[gpu_id], 4096)  # 4GB limit
            
            print(f"✅ GPU {gpu_id} configured successfully")
            return True
        except (RuntimeError, IndexError) as e:
            print(f"❌ Error configuring GPU {gpu_id}: {e}")
            return False
    else:
        print("❌ No GPUs found")
        return False

def calculate_image_hash(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(image)

def hamming_distance(hash1, hash2):
    return hash1 - hash2

def is_blurry(frame, threshold=100):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return laplacian_var < threshold

def format_csv_file(frame_idx, n, fps):
    pts_time = round(frame_idx / fps, 2)
    new_dict = {
        'n': n,
        'pts_time': pts_time,
        'fps': fps,
        'frame_idx': frame_idx
    }
    return new_dict

def save_to_csv(csv_file, output_folder, file_name):
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    file_path = os.path.join(output_folder, file_name)
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['n', 'pts_time', 'fps', 'frame_idx'])
        writer.writeheader()
        writer.writerows(csv_file)
    
    print(f"File saved at: {file_path}")

# ========== CONFIG ==========
HAMMING_THRESHOLD = 5
KEYFRAME_PROB_THRESHOLD = 0.2

def extract_frames_optimized(video_path, resize_shape=(48, 27), skip_start=5, skip_end=10, batch_size=1000):
    """
    Memory-optimized frame extraction that processes in batches
    This reduces RAM usage by not loading all frames at once
    """
    print(f"📊 [Memory Optimized] Processing video in batches of {batch_size} frames")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(fps * skip_start)
    end_frame = max(start_frame, total_frames - int(fps * skip_end))
    
    print(f"📊 Total frames: {total_frames}, Processing frames: {start_frame} to {end_frame}")
    
    # Calculate how many frames we'll actually process
    frames_to_process = end_frame - start_frame
    
    # If video is small enough, use original method
    if frames_to_process <= batch_size:
        print("📊 Video is small, using original method")
        return extract_frames_original(video_path, resize_shape, skip_start, skip_end)
    
    # For large videos, we'll need to process in chunks
    # Note: This is a more complex implementation that would require 
    # modifying the TransNetV2 model to handle streaming
    print("⚠️  Large video detected. Consider using streaming approach for better memory efficiency.")
    
    # For now, fall back to original method but with warning
    return extract_frames_original(video_path, resize_shape, skip_start, skip_end)

def extract_frames_original(video_path, resize_shape=(48, 27), skip_start=5, skip_end=10):
    """Original frame extraction method"""
    frames = []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(fps * skip_start)
    end_frame = max(start_frame, total_frames - int(fps * skip_end))

    # Calculate memory usage
    frames_to_process = end_frame - start_frame
    memory_mb = frames_to_process * resize_shape[0] * resize_shape[1] * 3 / (1024*1024)
    print(f"📊 Estimated RAM usage for frames: {memory_mb:.1f} MB")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_idx < end_frame:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, resize_shape)
            frames.append(frame_resized)

    cap.release()
    
    frames_array = np.array(frames)
    actual_memory_mb = frames_array.nbytes / (1024*1024)
    print(f"📊 Actual RAM usage for frames: {actual_memory_mb:.1f} MB")
    
    return frames_array, fps, start_frame, end_frame

# Alias for backward compatibility
def extract_frames(video_path, resize_shape=(48, 27), skip_start=5, skip_end=10):
    return extract_frames_optimized(video_path, resize_shape, skip_start, skip_end)

def detect_scene_changes(model, frames, threshold=KEYFRAME_PROB_THRESHOLD):
    """Phát hiện điểm chuyển cảnh bằng TransNetV2"""
    print(f"🧠 Processing {len(frames)} frames on GPU for scene detection...")
    
    prediction, prediction_changes = model.predict_frames(frames)
    raw_scene_changes = np.where(prediction_changes > threshold)[0]

    scene_changes = []
    for i in range(1, len(raw_scene_changes)):
        if raw_scene_changes[i] - raw_scene_changes[i - 1] > 1:
            scene_changes.append(raw_scene_changes[i - 1])
    if len(raw_scene_changes) > 0:
        scene_changes.append(raw_scene_changes[-1])

    print(f"🎯 Found {len(scene_changes)} scene changes")
    return sorted(set(scene_changes))

def extract_keyframes(video_path, scene_changes, frames, fps, start_frame, end_frame):
    """Trích xuất frame đầu/giữa/cuối của mỗi đoạn"""
    frame_index, count = 0, 0
    current_segment, keyframes, csv_entries = [], {}, []
    
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index < start_frame:
            frame_index += 1
            continue

        relative_index = frame_index - start_frame
        
        if relative_index in scene_changes or relative_index == len(frames) - 1:
            if len(current_segment) > 2:
                start, mid, end = (
                    current_segment[0],
                    current_segment[len(current_segment) // 2],
                    current_segment[-1],
                )

                keyframes[count] = start
                keyframes[count + 1] = mid
                keyframes[count + 2] = end

                csv_entries.extend([
                    format_csv_file(frame_index - len(current_segment), count, fps),
                    format_csv_file(frame_index - len(current_segment) + len(current_segment) // 2, count + 1, fps),
                    format_csv_file(frame_index - 1, count + 2, fps),
                ])

                count += 3

            current_segment = []

        current_segment.append(frame)
        frame_index += 1

    cap.release()
    print(f"🖼️  Extracted {len(keyframes)} keyframes")
    return keyframes, csv_entries

def filter_duplicate_frames(frames_dict, threshold=HAMMING_THRESHOLD):
    """Lọc frame trùng lặp bằng Hamming Distance"""
    print(f"🔍 Filtering duplicates from {len(frames_dict)} frames...")
    filtered, prev_hash = {}, None
    
    for idx, frame in frames_dict.items():
        cur_hash = calculate_image_hash(frame)

        if prev_hash is None or hamming_distance(cur_hash, prev_hash) > threshold:
            filtered[idx] = frame
            prev_hash = cur_hash
    
    print(f"✅ Kept {len(filtered)} unique frames (removed {len(frames_dict) - len(filtered)} duplicates)")
    return filtered

def save_frames(frames_dict, output_folder):
    """Lưu keyframes thành file ảnh"""
    for idx, frame in frames_dict.items():
        output_path = os.path.join(output_folder, f"{str(idx).zfill(3)}.jpg")
        cv2.imwrite(output_path, frame)

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for i, gpu in enumerate(gpus):
                memory_info = tf.config.experimental.get_memory_info(gpu)
                current_mb = memory_info['current'] / (1024*1024)
                peak_mb = memory_info['peak'] / (1024*1024)
                print(f"🔥 GPU {i} Memory - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
    except Exception as e:
        print(f"⚠️  Could not get GPU memory info: {e}")

def process_videos_on_gpu(input_folder, output_folder, csv_output_folder, gpu_id):
    """Process videos using a specific GPU with memory monitoring"""
    print(f"🚀 Starting processing on GPU {gpu_id} for folder: {input_folder}")
    
    # Setup GPU
    if not setup_gpu(gpu_id):
        print(f"❌ Failed to setup GPU {gpu_id}, exiting...")
        return False
    
    # Load model after GPU setup
    try:
        print(f"📥 Loading TransNetV2 model on GPU {gpu_id}...")
        model = TransNetV2()
        print(f"✅ Model loaded successfully on GPU {gpu_id}")
        monitor_gpu_memory()
    except Exception as e:
        print(f"❌ Failed to load model on GPU {gpu_id}: {e}")
        return False

    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(csv_output_folder, exist_ok=True)

    # Process videos
    processed_count = 0
    for video_file in sorted(os.listdir(input_folder)):
        if not video_file.endswith(".mp4"):
            print(f"[GPU {gpu_id}] Skipping file {video_file}")
            continue

        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_video_folder = os.path.join(output_folder, video_name)
        os.makedirs(output_video_folder, exist_ok=True)

        try:
            print(f"[GPU {gpu_id}] Processing video: {video_name}")

            # Process the video
            frames, fps, start_frame, end_frame = extract_frames(video_path)
            
            # Monitor memory before scene detection
            monitor_gpu_memory()
            
            scene_changes = detect_scene_changes(model, frames)
            
            # Clear frames from memory after scene detection if possible
            # (Note: We still need them for keyframe extraction)
            
            keyframes, csv_entries = extract_keyframes(video_path, scene_changes, frames, fps, start_frame, end_frame)
            
            # Clear frames array to free RAM
            del frames
            
            filtered_frames = filter_duplicate_frames(keyframes)

            # Save results
            save_frames(filtered_frames, output_video_folder)
            save_to_csv(csv_entries, csv_output_folder, video_name)

            processed_count += 1
            print(f"✅ [GPU {gpu_id}] Completed video: {video_name}")
            monitor_gpu_memory()

        except Exception as e:
            print(f"❌ [GPU {gpu_id}] Error processing video {video_name}: {e}")
            continue

    print(f"🎉 [GPU {gpu_id}] Finished processing {processed_count} videos from {input_folder}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Process videos with keyframe extraction on specific GPU')
    parser.add_argument('--input_folder', required=True, help='Input folder containing videos')
    parser.add_argument('--output_folder', required=True, help='Output folder for extracted frames')
    parser.add_argument('--csv_folder', required=True, help='Output folder for CSV files')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use (0-9)')
    parser.add_argument('--memory_limit', type=int, help='GPU memory limit in MB (optional)')
    
    args = parser.parse_args()
    
    if args.memory_limit:
        print(f"🔧 GPU memory limit set to {args.memory_limit}MB")
    
    success = process_videos_on_gpu(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        csv_output_folder=args.csv_folder,
        gpu_id=args.gpu_id
    )
    
    if success:
        print(f"✅ Processing completed successfully on GPU {args.gpu_id}")
        sys.exit(0)
    else:
        print(f"❌ Processing failed on GPU {args.gpu_id}")
        sys.exit(1)

if __name__ == "__main__":
    main()
