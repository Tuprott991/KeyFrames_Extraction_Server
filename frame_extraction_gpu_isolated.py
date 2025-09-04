import os
import sys
sys.path.append("/home/tuktu/KeyFrames_Extraction_Server/TransNetV2/inference")

# Suppress TensorFlow warnings and NUMA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from transnetv2 import TransNetV2
from video_decoder_utils import RobustVideoCapture, test_video_decode

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

# Global constants
RESIZE_SHAPE = (48, 27)
KEYFRAME_PROB_THRESHOLD = 0.5

def setup_gpu_isolated():
    """Setup GPU in isolated mode using CUDA_VISIBLE_DEVICES"""
    try:
        # Check what GPU is visible through environment variable
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        print(f"🔍 CUDA_VISIBLE_DEVICES: {visible_devices}")
        
        # List available GPUs (should only be one due to CUDA_VISIBLE_DEVICES)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if not gpus:
            print(f"❌ No GPUs found (CUDA_VISIBLE_DEVICES={visible_devices})")
            return False
            
        print(f"✅ Found {len(gpus)} GPU(s) in isolated mode:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
        
        # Configure memory growth for all visible GPUs
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Memory growth configured for all visible GPUs")
        except RuntimeError as e:
            print(f"⚠️  Memory growth configuration: {e}")
        
        # Test GPU functionality
        with tf.device('/GPU:0'):  # Use GPU 0 in isolated mode
            test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = tf.matmul(test_tensor, test_tensor)
            print(f"✅ GPU test operation successful")
                
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
        'frame_idx': frame_idx
    }
    return new_dict

def save_to_csv(csv_entries, csv_folder, video_name):
    csv_file = os.path.join(csv_folder, f"{video_name}.csv")
    os.makedirs(csv_folder, exist_ok=True)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        if csv_entries:
            fieldnames = csv_entries[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for entry in csv_entries:
                writer.writerow(entry)

def save_frames(keyframes, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for idx, (frame_idx, frame) in enumerate(keyframes.items()):
        filename = f"frame_{frame_idx:06d}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)

def extract_frames(video_path, skip_start=0, skip_end=0, resize_shape=RESIZE_SHAPE):
    """Extract frames from video with robust codec handling"""
    print(f"🎬 Extracting frames from: {os.path.basename(video_path)}")
    
    # Test video decode capability first
    if not test_video_decode(video_path, num_test_frames=5):
        print(f"❌ Video decode test failed for: {video_path}")
        return None, None, None, None
    
    frames = []
    
    with RobustVideoCapture(video_path) as cap:
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None, None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            print(f"❌ Invalid video properties: fps={fps}, frames={total_frames}")
            return None, None, None, None
        
        start_frame = int(fps * skip_start)
        end_frame = max(start_frame, total_frames - int(fps * skip_end))
        
        print(f"📊 Processing frames {start_frame} to {end_frame} (total: {total_frames})")
        
        frame_idx = 0
        successful_reads = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️  Frame read failed at index {frame_idx}")
                break
            
            if start_frame <= frame_idx < end_frame:
                if frame is not None and frame.size > 0:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, resize_shape)
                        frames.append(frame_resized)
                        successful_reads += 1
                    except Exception as e:
                        print(f"⚠️  Error processing frame {frame_idx}: {e}")
                        continue
            
            frame_idx += 1
            
            # Progress indicator for long videos
            if frame_idx % 1000 == 0:
                print(f"📈 Processed {frame_idx}/{total_frames} frames...")
        
        print(f"✅ Successfully extracted {successful_reads} frames")
        
        if len(frames) == 0:
            print(f"❌ No frames were successfully extracted")
            return None, None, None, None
            
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
    """Extract keyframes with robust codec handling"""
    frame_index, count = 0, 0
    current_segment, keyframes, csv_entries = [], {}, []
    
    with RobustVideoCapture(video_path) as cap:
        if not cap.isOpened():
            print(f"❌ Cannot reopen video file: {video_path}")
            return {}, []

        successful_reads = 0
        failed_reads = 0

        while frame_index < end_frame:
            ret, frame = cap.read()
            if not ret:
                if failed_reads > 10:  # Too many consecutive failures
                    print(f"⚠️  Too many failed reads ({failed_reads}), stopping keyframe extraction")
                    break
                failed_reads += 1
                frame_index += 1
                continue
            
            failed_reads = 0  # Reset on successful read
            successful_reads += 1

            if start_frame <= frame_index < end_frame:
                if frame_index in scene_changes:
                    # Process current segment
                    if current_segment:
                        segment_start = current_segment[0][0]
                        segment_end = current_segment[-1][0]
                        segment_mid = segment_start + (segment_end - segment_start) // 2

                        # Add start, middle, end frames
                        for pos_name, pos_idx in [("start", segment_start), ("middle", segment_mid), ("end", segment_end)]:
                            for seg_idx, seg_frame in current_segment:
                                if seg_idx == pos_idx:
                                    keyframes[seg_idx] = seg_frame
                                    csv_entries.append(format_csv_file(seg_idx, count, fps))
                                    count += 1
                                    break
                    
                    # Start new segment
                    current_segment = [(frame_index, frame)]
                else:
                    if len(current_segment) > 0:
                        current_segment.append((frame_index, frame))

            frame_index += 1

        # Process last segment
        if current_segment:
            segment_start = current_segment[0][0]
            segment_end = current_segment[-1][0]
            segment_mid = segment_start + (segment_end - segment_start) // 2

            for pos_name, pos_idx in [("start", segment_start), ("middle", segment_mid), ("end", segment_end)]:
                for seg_idx, seg_frame in current_segment:
                    if seg_idx == pos_idx:
                        keyframes[seg_idx] = seg_frame
                        csv_entries.append(format_csv_file(seg_idx, count, fps))
                        count += 1
                        break

    print(f"✅ Extracted {len(keyframes)} keyframes from {successful_reads} frames")
    return keyframes, csv_entries

def filter_duplicate_frames(keyframes):
    """Lọc bỏ frame trùng lặp dựa trên perceptual hash"""
    filtered_frames = {}
    seen_hashes = {}

    for frame_idx, frame in keyframes.items():
        frame_hash = calculate_image_hash(frame)
        
        is_duplicate = False
        for seen_hash in seen_hashes:
            if hamming_distance(frame_hash, seen_hash) <= 5:
                is_duplicate = True
                break
        
        if not is_duplicate and not is_blurry(frame):
            filtered_frames[frame_idx] = frame
            seen_hashes[frame_hash] = frame_idx

    return filtered_frames

def process_videos_isolated(input_folder, output_folder, csv_output_folder):
    """Process videos using isolated GPU setup"""
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"🚀 Starting isolated processing with CUDA_VISIBLE_DEVICES={visible_devices}")
    print(f"📁 Input folder: {input_folder}")
    
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
    
    # Setup GPU in isolated mode
    if not setup_gpu_isolated():
        print(f"❌ Failed to setup GPU in isolated mode")
        return False
    
    # Load model
    try:
        print(f"🔄 Loading TransNetV2 model...")
        with tf.device('/GPU:0'):  # Use GPU 0 in isolated mode
            model = TransNetV2()
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Create output folders
    try:
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(csv_output_folder, exist_ok=True)
        print(f"📂 Output folders created")
    except Exception as e:
        print(f"❌ Failed to create output folders: {e}")
        return False

    # Process videos
    processed_count = 0
    skipped_count = 0
    
    for video_file in sorted(video_files):
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_video_folder = os.path.join(output_folder, video_name)
        
        try:
            print(f"🎬 Processing video: {video_name}")

            # Process the video with robust codec handling
            result = extract_frames(video_path)
            if result[0] is None:  # Check if extraction failed
                print(f"❌ Failed to extract frames from {video_name}")
                skipped_count += 1
                continue
                
            frames, fps, start_frame, end_frame = result
            
            if len(frames) == 0:
                print(f"❌ No frames extracted from {video_name}")
                skipped_count += 1
                continue
                
            scene_changes = detect_scene_changes(model, frames)
            keyframes, csv_entries = extract_keyframes(video_path, scene_changes, frames, fps, start_frame, end_frame)
            
            if len(keyframes) == 0:
                print(f"⚠️  No keyframes detected in {video_name}")
                skipped_count += 1
                continue
                
            filtered_frames = filter_duplicate_frames(keyframes)

            # Save results
            save_frames(filtered_frames, output_video_folder)
            save_to_csv(csv_entries, csv_output_folder, video_name)

            processed_count += 1
            print(f"✅ Completed video: {video_name} ({len(filtered_frames)} frames)")

        except Exception as e:
            error_msg = str(e)
            if "codec" in error_msg.lower() or "av1" in error_msg.lower() or "sequence header" in error_msg.lower():
                print(f"❌ Codec error processing video {video_name}: {e}")
                print(f"💡 This video likely uses AV1, VP9, or another unsupported codec")
                print(f"💡 Try: python video_decoder_utils.py {video_path}")
            else:
                print(f"❌ Error processing video {video_name}: {e}")
            skipped_count += 1
            continue

    print(f"🎉 Processing complete!")
    print(f"   ✅ Successfully processed: {processed_count}/{len(video_files)} videos")
    print(f"   ⚠️  Skipped due to codec issues: {skipped_count}/{len(video_files)} videos")
    
    if skipped_count > 0:
        print(f"\n💡 To fix codec issues, run:")
        print(f"   python video_codec_fix.py --input_folder {input_folder}")
    
    return processed_count > 0

def main():
    parser = argparse.ArgumentParser(description='Process videos with keyframe extraction in isolated GPU mode')
    parser.add_argument('--input_folder', required=True, help='Input folder containing videos')
    parser.add_argument('--output_folder', required=True, help='Output folder for extracted frames')
    parser.add_argument('--csv_folder', required=True, help='Output folder for CSV files')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID (for logging purposes)')
    
    args = parser.parse_args()
    
    success = process_videos_isolated(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        csv_output_folder=args.csv_folder
    )
    
    if success:
        print(f"✅ Processing completed successfully")
        sys.exit(0)
    else:
        print(f"❌ Processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
