#!/usr/bin/env python3
"""
Video Codec Fix Utility
Handles AV1 and other codec issues by converting problematic videos to H.264
"""

import os
import sys
import subprocess
import cv2
import shutil
import tempfile
from pathlib import Path

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
        else:
            print("❌ FFmpeg not found or not working")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not found in PATH")
        return False

def check_available_encoders():
    """Check which hardware encoders are available"""
    available_encoders = []
    
    encoders_to_check = [
        ('h264_nvenc', 'NVIDIA NVENC'),
        ('h264_vaapi', 'Intel/AMD VAAPI'),
        ('h264_qsv', 'Intel QuickSync'),
        ('libx264', 'CPU x264')
    ]
    
    for encoder, description in encoders_to_check:
        try:
            # Test if encoder is available
            result = subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1',
                '-c:v', encoder, '-f', 'null', '-'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                available_encoders.append((encoder, description))
                print(f"✅ {description} ({encoder}) is available")
            else:
                print(f"❌ {description} ({encoder}) not available")
                
        except Exception:
            print(f"❌ {description} ({encoder}) not available")
    
    return available_encoders

def get_video_info(video_path):
    """Get video codec and other information"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', '-select_streams', 'v:0', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                codec = stream.get('codec_name', 'unknown')
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                duration = stream.get('duration', '0')
                return {
                    'codec': codec,
                    'width': width,
                    'height': height,
                    'duration': float(duration),
                    'valid': True
                }
    except Exception as e:
        print(f"⚠️  Could not get video info for {video_path}: {e}")
    
    return {'valid': False}

def test_opencv_read(video_path):
    """Test if OpenCV can read the video"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Try to read a few frames
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False
        
        cap.release()
        return True
    except Exception:
        return False

def convert_video_to_h264(input_path, output_path, gpu_accelerated=True):
    """Convert video to H.264 codec for better compatibility"""
    try:
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if gpu_accelerated:
            # Try different GPU encoders in order of preference
            gpu_encoders = [
                {
                    'name': 'h264_nvenc',
                    'cmd': [
                        'ffmpeg', '-y', '-i', input_path,
                        '-c:v', 'h264_nvenc',  # NVIDIA NVENC
                        '-preset', 'fast',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-movflags', '+faststart',
                        output_path
                    ]
                },
                {
                    'name': 'h264_vaapi',
                    'cmd': [
                        'ffmpeg', '-y', '-hwaccel', 'vaapi', '-hwaccel_device', '/dev/dri/renderD128',
                        '-i', input_path,
                        '-c:v', 'h264_vaapi',  # Intel/AMD VAAPI
                        '-qp', '23',
                        '-c:a', 'aac',
                        '-movflags', '+faststart',
                        output_path
                    ]
                },
                {
                    'name': 'h264_qsv',
                    'cmd': [
                        'ffmpeg', '-y', '-hwaccel', 'qsv', '-i', input_path,
                        '-c:v', 'h264_qsv',  # Intel QuickSync
                        '-preset', 'fast',
                        '-global_quality', '23',
                        '-c:a', 'aac',
                        '-movflags', '+faststart',
                        output_path
                    ]
                }
            ]
            
            for encoder in gpu_encoders:
                print(f"🔄 Trying GPU encoder: {encoder['name']} for {os.path.basename(input_path)}...")
                try:
                    result = subprocess.run(encoder['cmd'], capture_output=True, text=True, timeout=600)
                    
                    if result.returncode == 0:
                        print(f"✅ Successfully converted with {encoder['name']}")
                        return True
                    else:
                        print(f"⚠️  {encoder['name']} failed: {result.stderr.strip()}")
                        continue
                        
                except subprocess.TimeoutExpired:
                    print(f"⚠️  {encoder['name']} timeout")
                    continue
                except Exception as e:
                    print(f"⚠️  {encoder['name']} error: {e}")
                    continue
            
            print("🔄 All GPU encoders failed, falling back to CPU encoding...")
        
        # CPU encoding fallback
        cpu_cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264',  # CPU encoder
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ]
        
        print(f"🔄 Converting {os.path.basename(input_path)} with CPU encoding...")
        result = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=1200)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted with CPU encoding")
            return True
        else:
            print(f"❌ CPU conversion failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Conversion timeout for {input_path}")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def fix_videos_in_folder(input_folder, replace_original=True):
    """Fix all problematic videos in a folder"""
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        return False
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        print("❌ FFmpeg is required for video conversion")
        return False
    
    # Check available encoders
    print("\n🔍 Checking available hardware encoders...")
    available_encoders = check_available_encoders()
    
    if not available_encoders:
        print("❌ No video encoders available")
        return False
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        video_files.extend(Path(input_folder).glob(f"*{ext}"))
        video_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ No video files found in {input_folder}")
        return False
    
    print(f"🔍 Found {len(video_files)} video files")
    
    problematic_videos = []
    working_videos = []
    
    # Test each video
    for video_path in video_files:
        print(f"\n🎬 Testing: {video_path.name}")
        
        # Get video info
        info = get_video_info(str(video_path))
        if info['valid']:
            print(f"   Codec: {info['codec']}, Resolution: {info['width']}x{info['height']}")
            
            # Check if codec is problematic
            problematic_codecs = ['av1', 'vp9', 'hevc', 'h265']
            if info['codec'].lower() in problematic_codecs:
                print(f"   ⚠️  Problematic codec detected: {info['codec']}")
                problematic_videos.append(video_path)
                continue
        
        # Test OpenCV compatibility
        if test_opencv_read(str(video_path)):
            print(f"   ✅ OpenCV can read this video")
            working_videos.append(video_path)
        else:
            print(f"   ❌ OpenCV cannot read this video")
            problematic_videos.append(video_path)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Working videos: {len(working_videos)}")
    print(f"   ❌ Problematic videos: {len(problematic_videos)}")
    
    if problematic_videos:
        print(f"\n🔧 Converting {len(problematic_videos)} problematic videos...")
        
        converted_count = 0
        
        for video_path in problematic_videos:
            # Create temporary file for conversion
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                if convert_video_to_h264(str(video_path), temp_path):
                    # Test the converted video
                    if test_opencv_read(temp_path):
                        print(f"   ✅ Converted video works with OpenCV")
                        
                        if replace_original:
                            # Replace original file
                            shutil.move(temp_path, str(video_path))
                            print(f"   ✅ Replaced original file: {video_path.name}")
                        else:
                            # Keep both files
                            new_name = video_path.stem + "_fixed" + video_path.suffix
                            new_path = video_path.parent / new_name
                            shutil.move(temp_path, str(new_path))
                            print(f"   ✅ Saved fixed version: {new_name}")
                        
                        converted_count += 1
                    else:
                        print(f"   ⚠️  Converted video still has issues")
                        os.unlink(temp_path)  # Clean up temp file
                else:
                    print(f"   ❌ Failed to convert {video_path.name}")
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)  # Clean up temp file
                        
            except Exception as e:
                print(f"   ❌ Error processing {video_path.name}: {e}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)  # Clean up temp file
        
        print(f"\n🎉 Successfully converted {converted_count}/{len(problematic_videos)} videos")
        
        return converted_count > 0
    else:
        print("\n🎉 All videos are compatible with OpenCV!")
        return True

def process_all_video_folders(base_path="/home/tuktu/KeyFrames_Extraction_Server", replace_original=True):
    """Process all Videos_K01 to Videos_K20 folders"""
    
    total_processed = 0
    total_converted = 0
    
    for k in range(1, 21):  # K01 to K20
        folder_name = f"Videos_K{k:02d}"
        video_folder = os.path.join(base_path, folder_name, "video")
        
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"Path: {video_folder}")
        print(f"{'='*60}")
        
        if not os.path.exists(video_folder):
            print(f"⚠️  Folder does not exist: {video_folder}")
            continue
        
        try:
            success = fix_videos_in_folder(video_folder, replace_original)
            if success:
                total_processed += 1
                print(f"✅ Successfully processed {folder_name}")
            else:
                print(f"⚠️  Issues found in {folder_name}")
                
        except Exception as e:
            print(f"❌ Error processing {folder_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"🎉 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total folders processed: {total_processed}/20")
    print(f"{'='*60}")
    
    return total_processed > 0

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix video codec issues for OpenCV processing using FFmpeg with L4 GPU')
    parser.add_argument('--input_folder', help='Single input folder containing videos')
    parser.add_argument('--test_only', action='store_true', help='Only test videos, do not convert')
    parser.add_argument('--process_all', action='store_true', help='Process all Videos_K01 to Videos_K20 folders')
    parser.add_argument('--base_path', default='/home/tuktu/KeyFrames_Extraction_Server', 
                       help='Base path containing Videos_K01 to Videos_K20 folders')
    parser.add_argument('--keep_original', action='store_true', 
                       help='Keep original files (create _fixed versions instead of replacing)')
    
    args = parser.parse_args()
    
    if args.process_all:
        # Process all Videos_K01 to Videos_K20 folders
        print("🚀 Processing all Videos_K01 to Videos_K20 folders with L4 GPU acceleration...")
        success = process_all_video_folders(args.base_path, replace_original=not args.keep_original)
        sys.exit(0 if success else 1)
    elif args.input_folder:
        if args.test_only:
            # Just test the videos
            video_files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                video_files.extend(Path(args.input_folder).glob(f"*{ext}"))
            
            for video_path in video_files:
                info = get_video_info(str(video_path))
                opencv_ok = test_opencv_read(str(video_path))
                print(f"{video_path.name}: Codec={info.get('codec', 'unknown')}, OpenCV={'✅' if opencv_ok else '❌'}")
        else:
            success = fix_videos_in_folder(args.input_folder, replace_original=not args.keep_original)
            sys.exit(0 if success else 1)
    else:
        # Default: process all folders
        print("🚀 No specific folder provided. Processing all Videos_K01 to Videos_K20 folders with L4 GPU acceleration...")
        success = process_all_video_folders(args.base_path, replace_original=not args.keep_original)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    # If run without arguments, process all folders by default
    if len(sys.argv) == 1:
        print("🚀 Processing all Videos_K01 to Videos_K20 folders with L4 GPU acceleration...")
        success = process_all_video_folders()
        sys.exit(0 if success else 1)
    else:
        main()
