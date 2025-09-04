#!/usr/bin/env python3
"""
Video Codec Fix Utility
Handles AV1 and other codec issues by converting problematic videos to H.264
"""

import os
import sys
import subprocess
import cv2
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
            # Try GPU-accelerated encoding first (NVENC for NVIDIA)
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'h264_nvenc',  # NVIDIA GPU encoder
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                output_path
            ]
        else:
            # Fallback to CPU encoding
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-c:v', 'libx264',  # CPU encoder
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                output_path
            ]
        
        print(f"🔄 Converting {os.path.basename(input_path)} to H.264...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted to {output_path}")
            return True
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            if gpu_accelerated:
                print("🔄 Retrying with CPU encoding...")
                return convert_video_to_h264(input_path, output_path, gpu_accelerated=False)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Conversion timeout for {input_path}")
        return False
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False

def fix_videos_in_folder(input_folder, output_folder=None):
    """Fix all problematic videos in a folder"""
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        return False
    
    if output_folder is None:
        output_folder = os.path.join(input_folder, "converted")
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        print("❌ FFmpeg is required for video conversion")
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
            problematic_codecs = ['av1', 'vp9', 'hevc']
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
        
        os.makedirs(output_folder, exist_ok=True)
        converted_count = 0
        
        for video_path in problematic_videos:
            output_path = os.path.join(output_folder, f"fixed_{video_path.name}")
            
            if convert_video_to_h264(str(video_path), output_path):
                # Test the converted video
                if test_opencv_read(output_path):
                    print(f"   ✅ Converted video works with OpenCV")
                    converted_count += 1
                else:
                    print(f"   ⚠️  Converted video still has issues")
            else:
                print(f"   ❌ Failed to convert {video_path.name}")
        
        print(f"\n🎉 Successfully converted {converted_count}/{len(problematic_videos)} videos")
        print(f"📁 Converted videos saved to: {output_folder}")
        
        return converted_count > 0
    else:
        print("\n🎉 All videos are compatible with OpenCV!")
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix video codec issues for OpenCV processing')
    parser.add_argument('--input_folder', required=True, help='Input folder containing videos')
    parser.add_argument('--output_folder', help='Output folder for converted videos')
    parser.add_argument('--test_only', action='store_true', help='Only test videos, do not convert')
    
    args = parser.parse_args()
    
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
        success = fix_videos_in_folder(args.input_folder, args.output_folder)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
