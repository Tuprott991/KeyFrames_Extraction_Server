import os
import sys
sys.path.append("/home/tuktu/KeyFrames_Extraction_Server/TransNetV2/inference")

# Suppress TensorFlow warnings and NUMA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from transnetv2 import TransNetV2

import numpy as np
import tensorflow as tf
import cv2
import json
import imagehash
from PIL import Image
import csv
import sys
import argparse

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')

def setup_gpu(gpu_id):
    """Configure TensorFlow to use a specific GPU with improved error handling"""
    try:
        # List available GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if not gpus:
            print(f"❌ No GPUs found on the system")
            return False
            
        if gpu_id >= len(gpus):
            print(f"❌ GPU {gpu_id} not available. Available GPUs: {len(gpus)} (0-{len(gpus)-1})")
            return False
        
        print(f"🔍 Available GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
        
        # Configure memory growth for all GPUs (must be done before any operations)
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memory growth configured for all GPUs")
        except RuntimeError as e:
            if "Memory growth" in str(e):
                print(f"⚠️  Memory growth already configured: {e}")
            else:
                print(f"⚠️  Could not configure memory growth: {e}")
        
        # Test GPU functionality
        try:
            with tf.device(f'/GPU:{gpu_id}'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                print(f"✅ GPU {gpu_id} test operation successful")
        except Exception as e:
            print(f"❌ GPU {gpu_id} test operation failed: {e}")
            return False
                
        return True
                
    except Exception as e:
        print(f"❌ Unexpected error during GPU setup: {e}")
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

def extract_frames(video_path, resize_shape=(48, 27), skip_start=5, skip_end=10):
    """Đọc toàn bộ frame từ video và resize để dùng cho TransNetV2, bỏ qua skip_start giây đầu và skip_end giây cuối"""
    frames = []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(fps * skip_start)
    end_frame = max(start_frame, total_frames - int(fps * skip_end))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if start_frame <= frame_idx < end_frame:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, resize_shape)
            frames.append(frame_resized)

    cap.release()
    return np.array(frames), fps, start_frame, end_frame

def detect_scene_changes(model, frames, threshold=KEYFRAME_PROB_THRESHOLD):
    """Phát hiện điểm chuyển cảnh bằng TransNetV2"""
    prediction, prediction_changes = model.predict_frames(frames)
    raw_scene_changes = np.where(prediction_changes > threshold)[0]

    scene_changes = []
    for i in range(1, len(raw_scene_changes)):
        if raw_scene_changes[i] - raw_scene_changes[i - 1] > 1:
            scene_changes.append(raw_scene_changes[i - 1])
    if len(raw_scene_changes) > 0:
        scene_changes.append(raw_scene_changes[-1])

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
    return keyframes, csv_entries

def filter_duplicate_frames(frames_dict, threshold=HAMMING_THRESHOLD):
    """Lọc frame trùng lặp bằng Hamming Distance"""
    filtered, prev_hash = {}, None
    
    for idx, frame in frames_dict.items():
        cur_hash = calculate_image_hash(frame)

        if prev_hash is None or hamming_distance(cur_hash, prev_hash) > threshold:
            filtered[idx] = frame
            prev_hash = cur_hash
            
    return filtered

def save_frames(frames_dict, output_folder):
    """Lưu keyframes thành file ảnh"""
    for idx, frame in frames_dict.items():
        output_path = os.path.join(output_folder, f"{str(idx).zfill(3)}.jpg")
        cv2.imwrite(output_path, frame)

def process_videos_on_gpu(input_folder, output_folder, csv_output_folder, gpu_id):
    """Process videos using a specific GPU with improved error handling"""
    print(f"🚀 Starting processing on GPU {gpu_id} for folder: {input_folder}")
    
    # Validate input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        return False
    
    # Check if there are any videos to process
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    if not video_files:
        print(f"❌ No MP4 videos found in {input_folder}")
        return False
    
    print(f"📁 Found {len(video_files)} videos to process")
    
    # Setup GPU
    if not setup_gpu(gpu_id):
        print(f"❌ Failed to setup GPU {gpu_id}, exiting...")
        return False
    
    # Load model after GPU setup
    try:
        print(f"🔄 Loading TransNetV2 model on GPU {gpu_id}...")
        with tf.device(f'/GPU:{gpu_id}'):
            model = TransNetV2()
        print(f"✅ Model loaded successfully on GPU {gpu_id}")
    except Exception as e:
        print(f"❌ Failed to load model on GPU {gpu_id}: {e}")
        print(f"💡 This might be due to GPU memory issues or model file problems")
        return False

    # Create output folders
    try:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(csv_output_folder, exist_ok=True)
        print(f"📂 Output folders created: {output_folder}, {csv_output_folder}")
    except Exception as e:
        print(f"❌ Failed to create output folders: {e}")
        return False

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
            scene_changes = detect_scene_changes(model, frames)
            keyframes, csv_entries = extract_keyframes(video_path, scene_changes, frames, fps, start_frame, end_frame)
            filtered_frames = filter_duplicate_frames(keyframes)

            # Save results
            save_frames(filtered_frames, output_video_folder)
            save_to_csv(csv_entries, csv_output_folder, video_name)

            processed_count += 1
            print(f"✅ [GPU {gpu_id}] Completed video: {video_name}")

        except Exception as e:
            print(f"❌ [GPU {gpu_id}] Error processing video {video_name}: {e}")
            continue

    print(f"🎉 [GPU {gpu_id}] Finished processing {processed_count} videos from {input_folder}")
    return True

# Print all available GPU devices
def print_gpu_info():
    """Print comprehensive GPU information"""
    try:
        print("🔍 Checking GPU availability...")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if not gpus:
            print("❌ No GPU devices found")
            return
            
        print(f"✅ Found {len(gpus)} GPU device(s):")
        for i, gpu in enumerate(gpus):
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"   GPU {i}: {gpu}")
                if 'device_name' in gpu_details:
                    print(f"       Name: {gpu_details['device_name']}")
                if 'compute_capability' in gpu_details:
                    print(f"       Compute Capability: {gpu_details['compute_capability']}")
            except:
                print(f"   GPU {i}: {gpu}")
                
        # Check if CUDA is available
        print(f"🎯 CUDA available: {tf.test.is_built_with_cuda()}")
        print(f"🎯 GPU support: {tf.test.is_gpu_available()}")
        
    except Exception as e:
        print(f"❌ Error checking GPU info: {e}")

# Call GPU info function
print_gpu_info()

def main():
    parser = argparse.ArgumentParser(description='Process videos with keyframe extraction on specific GPU')
    parser.add_argument('--input_folder', required=True, help='Input folder containing videos')
    parser.add_argument('--output_folder', required=True, help='Output folder for extracted frames')
    parser.add_argument('--csv_folder', required=True, help='Output folder for CSV files')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use (0-9)')
    
    args = parser.parse_args()
    
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
