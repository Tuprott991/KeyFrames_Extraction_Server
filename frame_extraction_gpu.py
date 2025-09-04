import os
import sys
sys.path.append("/home/tuktu/KeyFrames_Extraction_Server/TransNetV2")

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

def setup_gpu(gpu_id):
    """Configure TensorFlow to use a specific GPU"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for the specified GPU
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
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
    """Process videos using a specific GPU"""
    print(f"🚀 Starting processing on GPU {gpu_id} for folder: {input_folder}")
    
    # Setup GPU
    if not setup_gpu(gpu_id):
        print(f"❌ Failed to setup GPU {gpu_id}, exiting...")
        return False
    
    # Load model after GPU setup
    try:
        model = TransNetV2()
        print(f"✅ Model loaded successfully on GPU {gpu_id}")
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
