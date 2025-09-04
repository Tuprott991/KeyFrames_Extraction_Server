from TransNetV2.inference.transnetv2 import TransNetV2
import os
import numpy as np
import tensorflow as tf
import cv2
import json
import imagehash
from PIL import Image
import csv

def calculate_image_hash(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Chuyển frame thành định dạng ảnh
    return imagehash.phash(image)  # Sử dụng phash để tính toán hash của hình ảnh

# Hàm tính khoảng cách Hamming giữa hai giá trị hash
def hamming_distance(hash1, hash2):
    return hash1 - hash2

# ...
def is_blurry(frame, threshold=100):
    # Chuyển đổi ảnh sang grayscale (ảnh xám)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tính toán độ biến thiên của Laplacian
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    
    # Nếu độ biến thiên thấp hơn ngưỡng threshold, coi như frame bị mờ
    return laplacian_var < threshold

# ...
def format_csv_file(frame_idx, n, fps):
    pts_time = round(frame_idx / fps, 2)
    new_dict = {
        'n': n,
        'pts_time': pts_time,
        'fps': fps,
        'frame_idx': frame_idx
    }
    return new_dict

# ..
def save_to_csv(csv_file, output_folder, file_name):
    # Đảm bảo file_name có phần mở rộng .csv
    if not file_name.endswith('.csv'):
        file_name += '.csv'
    
    # Đảm bảo thư mục output_folder tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Đường dẫn đầy đủ tới file
    file_path = os.path.join(output_folder, file_name)
    
    # Mở file csv và ghi nội dung
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['n', 'pts_time', 'fps', 'frame_idx'])
        writer.writeheader()
        writer.writerows(csv_file)
    
    print(f"File saved at: {file_path}")

# ========== CONFIG ==========
# Ngưỡng lọc trùng
HAMMING_THRESHOLD = 5
# Ngưỡng chuyển cảnh
KEYFRAME_PROB_THRESHOLD = 0.2
# Load model
model = TransNetV2()    

def extract_frames(video_path, resize_shape=(48, 27), skip_start=5, skip_end=10):
    """Đọc toàn bộ frame từ video và resize để dùng cho TransNetV2, bỏ qua skip_start giây đầu và skip_end giây cuối"""
    frames = [] # Mảng lưu các frame tiền xử lý
    
    # Đọc video
    cap = cv2.VideoCapture(video_path)

    # Lấy frame ở skip_start giây đầu và skip_end giây cuối
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(fps * skip_start)
    end_frame = max(start_frame, total_frames - int(fps * skip_end))  # tránh âm

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break # Kết thúc nếu không còn frame nào để đọc

        if start_frame <= frame_idx < end_frame:
            # Chuyển đổi màu sắc từ BGR sang RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Đổi kích thước frame để phục vụ cho việc dự đoán
            frame_resized = cv2.resize(frame_rgb, resize_shape)
            # Thêm frame đã thay đổi vào mảng
            frames.append(frame_resized)

    # Giải phóng video đã đọc
    cap.release()
    
    # Chuyển toàn bộ frame về mảng Numpy rồi trả về
    return np.array(frames), fps, start_frame, end_frame


def detect_scene_changes(model, frames, threshold=KEYFRAME_PROB_THRESHOLD):
    """Phát hiện điểm chuyển cảnh bằng TransNetV2"""
    # prediction: xác suất cho từng frame.
    # prediction_changes: xác suất thay đổi cảnh giữa 2 frame liên tiếp.
    prediction, prediction_changes = model.predict_frames(frames)

    # Tìm chỉ số xác suất thay đổi cảnh giữa 2 frame liên tiếp lớn hơn ngưỡng
    # Nó sẽ lưu frame mà thay đổi nhiều so với frame trước đó.
    raw_scene_changes = np.where(prediction_changes > threshold)[0]

    # Lọc nhiễu (bỏ các thay đổi liên tiếp nhau)
    scene_changes = []
    for i in range(1, len(raw_scene_changes)):
        if raw_scene_changes[i] - raw_scene_changes[i - 1] > 1:
            scene_changes.append(raw_scene_changes[i - 1])
    if len(raw_scene_changes) > 0:
        # Kiểm tra cả điểm chuyển cảnh cuối cùng sau khi kết thúc vòng lặp
        scene_changes.append(raw_scene_changes[-1])

    # Xóa các chỉ số cảnh mà trùng lặp và sắp xếp lại theo thứ tự tăng dần rồi trả về
    return sorted(set(scene_changes))


def extract_keyframes(video_path, scene_changes, frames, fps, start_frame, end_frame):
    """Trích xuất frame đầu/giữa/cuối của mỗi đoạn"""
    # count: Biến đếm lưu tên file
    frame_index, count = 0, 0
    # csv_entries: File map-keyframe
    current_segment, keyframes, csv_entries = [], {}, []
    
    # Đọc video
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Kết thúc nếu không còn frame nào để đọc

        # Bỏ qua vài giây đầu
        if frame_index < start_frame:
            frame_index += 1
            continue

        # Tính frame_index tương ứng với frames đã skip
        relative_index = frame_index - start_frame
        
        # Nếu đến điểm chuyển cảnh hoặc frame cuối
        if relative_index in scene_changes or relative_index == len(frames) - 1:
            # Nếu có trên 2 khung hình trong list thì lấy đầu giữa và cuối, sau đó reset list
            if len(current_segment) > 2:
                # Lấy khung hình đầu, giữa và cuối
                start, mid, end = (
                    current_segment[0],
                    current_segment[len(current_segment) // 2],
                    current_segment[-1],
                )

                keyframes[count] = start
                keyframes[count + 1] = mid
                keyframes[count + 2] = end

                # Thêm dữ liệu vào file map-keyframe
                csv_entries.extend([
                    format_csv_file(frame_index - len(current_segment), count, fps),
                    format_csv_file(frame_index - len(current_segment) + len(current_segment) // 2, count + 1, fps),
                    format_csv_file(frame_index - 1, count + 2, fps),
                ])

                count += 3

            # Reset danh sách khung hình hiện tại
            current_segment = []

        # Thêm khung hình hiện tại vào danh sách current_frames
        current_segment.append(frame)
        frame_index += 1

    # Giải phóng video đã đọc
    cap.release()
    
    return keyframes, csv_entries


def filter_duplicate_frames(frames_dict, threshold=HAMMING_THRESHOLD):
    """Lọc frame trùng lặp bằng Hamming Distance"""
    filtered, prev_hash = {}, None
    
    for idx, frame in frames_dict.items():
        # Tính hash của frame hiện tại
        cur_hash = calculate_image_hash(frame)

        # Nếu đây là frame đầu tiên hoặc khoảng cách Hamming lớn hơn ngưỡng
        if prev_hash is None or hamming_distance(cur_hash, prev_hash) > threshold:
            # Lưu frame vào kết quả sau khi lọc
            filtered[idx] = frame
            # Cập nhật hash của frame hiện tại
            prev_hash = cur_hash
            
    return filtered


def save_frames(frames_dict, output_folder):
    """Lưu keyframes thành file ảnh"""
    for idx, frame in frames_dict.items():
        output_path = os.path.join(output_folder, f"{str(idx).zfill(3)}.jpg")
        cv2.imwrite(output_path, frame)


# ========== MAIN PIPELINE ==========
def process_videos(input_folder, output_folder, csv_output_folder, model):
    # Tạo các folder output nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(csv_output_folder, exist_ok=True)

    # Duyệt qua tất cả các file trong thư mục input
    for video_file in sorted(os.listdir(input_folder)):
        if not video_file.endswith(".mp4") or video_file == "L25_V012.mp4":
            print(f"Bỏ qua file {video_file}")
            continue

        # Tạo đường dẫn video
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # Lấy tên video không bao gồm phần mở rộng
        output_video_folder = os.path.join(output_folder, video_name)
        os.makedirs(output_video_folder, exist_ok=True)

        try:
            print(f"Đang xử lý video: {video_name}")

            # Extract frames
            frames, fps, start_frame, end_frame = extract_frames(video_path)

            # Scene detection (using model in config)
            scene_changes = detect_scene_changes(model, frames)

            # Keyframe extraction (xuất keyframes)
            keyframes, csv_entries = extract_keyframes(video_path, scene_changes, frames, fps, start_frame, end_frame)

            # Remove duplicates (lọc ảnh trùng)
            filtered_frames = filter_duplicate_frames(keyframes)

            # Save results
            save_frames(filtered_frames, output_video_folder)
            save_to_csv(csv_entries, csv_output_folder, video_name)

            print(f"✅ Xử lý xong video: {video_name}")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý video {video_name}: {e}")
            continue

process_videos(
    input_folder="/kaggle/input/data-video-batch2-2/Videos_K19/video",
    output_folder="/kaggle/working/images",
    csv_output_folder="/kaggle/working/csv",
    model=model
)