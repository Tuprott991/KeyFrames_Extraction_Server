import cv2
import os
import subprocess
import tempfile
import numpy as np

def get_video_info(video_path):
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    return {
                        'codec': stream.get('codec_name', 'unknown'),
                        'width': stream.get('width', 0),
                        'height': stream.get('height', 0),
                        'fps': eval(stream.get('r_frame_rate', '0/1')),
                        'duration': float(stream.get('duration', 0))
                    }
    except Exception as e:
        print(f"⚠️  Could not get video info: {e}")
    return None

def is_problematic_codec(video_path):
    """Check if video uses problematic codec that needs software decoding"""
    info = get_video_info(video_path)
    if info:
        problematic_codecs = ['av1', 'vp9', 'hevc']
        codec = info.get('codec', '').lower()
        return codec in problematic_codecs
    return False

def convert_video_for_opencv(input_path, output_path=None):
    """Convert problematic video to H.264 for better OpenCV compatibility"""
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted.mp4"
    
    print(f"🔄 Converting {os.path.basename(input_path)} to H.264 for better compatibility...")
    
    # FFmpeg command with software decoding and H.264 encoding
    cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        '-hwaccel', 'none',  # Force software decoding
        '-i', input_path,
        '-c:v', 'libx264',  # Use H.264 codec
        '-preset', 'fast',  # Fast encoding
        '-crf', '23',  # Good quality
        '-pix_fmt', 'yuv420p',  # Compatible pixel format
        '-movflags', '+faststart',  # Web optimization
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Video converted successfully: {output_path}")
            return output_path
        else:
            print(f"❌ Conversion failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return None

def open_video_with_fallback(video_path):
    """Open video with fallback options for problematic codecs"""
    print(f"🎬 Opening video: {os.path.basename(video_path)}")
    
    # Try 1: Direct OpenCV open
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        # Test if we can actually read frames
        ret, frame = cap.read()
        if ret and frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            print(f"✅ Direct OpenCV open successful")
            return cap, video_path
        else:
            cap.release()
    
    print(f"⚠️  Direct OpenCV open failed, checking codec...")
    
    # Try 2: Check if it's a problematic codec and convert
    if is_problematic_codec(video_path):
        print(f"🔧 Detected problematic codec, converting...")
        converted_path = convert_video_for_opencv(video_path)
        if converted_path and os.path.exists(converted_path):
            cap = cv2.VideoCapture(converted_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print(f"✅ Converted video opened successfully")
                    return cap, converted_path
                else:
                    cap.release()
    
    # Try 3: Force different backends
    backends = [
        cv2.CAP_FFMPEG,
        cv2.CAP_GSTREAMER,
        cv2.CAP_V4L2,
        cv2.CAP_ANY
    ]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print(f"✅ Opened with backend {backend}")
                    return cap, video_path
                else:
                    cap.release()
        except Exception as e:
            continue
    
    print(f"❌ All video opening methods failed for {video_path}")
    return None, None

def safe_video_read(cap, max_retries=3):
    """Safely read video frame with retries"""
    for attempt in range(max_retries):
        try:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return ret, frame
            elif not ret:
                return False, None
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"❌ Frame read failed after {max_retries} attempts: {e}")
                return False, None
            continue
    return False, None

def cleanup_converted_video(video_path, original_path):
    """Clean up converted video file if it was created"""
    if video_path != original_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
            print(f"🗑️  Cleaned up converted file: {video_path}")
        except Exception as e:
            print(f"⚠️  Could not clean up converted file: {e}")

class RobustVideoCapture:
    """A robust video capture class that handles problematic codecs"""
    
    def __init__(self, video_path):
        self.original_path = video_path
        self.cap = None
        self.current_path = None
        self.frame_count = 0
        self.fps = 0
        self._initialize()
    
    def _initialize(self):
        """Initialize video capture with fallback options"""
        self.cap, self.current_path = open_video_with_fallback(self.original_path)
        if self.cap:
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"📊 Video info: {self.frame_count} frames, {self.fps:.2f} fps")
    
    def isOpened(self):
        """Check if video is opened"""
        return self.cap is not None and self.cap.isOpened()
    
    def read(self):
        """Read next frame with error handling"""
        if not self.isOpened():
            return False, None
        return safe_video_read(self.cap)
    
    def get(self, prop):
        """Get video property"""
        if not self.isOpened():
            return 0
        return self.cap.get(prop)
    
    def set(self, prop, value):
        """Set video property"""
        if not self.isOpened():
            return False
        return self.cap.set(prop, value)
    
    def release(self):
        """Release video capture and cleanup"""
        if self.cap:
            self.cap.release()
        # Clean up converted file if it was created
        cleanup_converted_video(self.current_path, self.original_path)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

def test_video_decode(video_path, num_test_frames=10):
    """Test if video can be decoded properly"""
    print(f"🧪 Testing video decode: {os.path.basename(video_path)}")
    
    with RobustVideoCapture(video_path) as cap:
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        successful_reads = 0
        for i in range(min(num_test_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if ret and frame is not None:
                successful_reads += 1
            else:
                break
        
        success_rate = successful_reads / num_test_frames if num_test_frames > 0 else 0
        print(f"📊 Decode test: {successful_reads}/{num_test_frames} frames ({success_rate*100:.1f}%)")
        
        return success_rate > 0.8  # Consider successful if >80% frames read

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            test_video_decode(video_path)
        else:
            print(f"Video file not found: {video_path}")
    else:
        print("Usage: python video_decoder_utils.py <video_path>")
