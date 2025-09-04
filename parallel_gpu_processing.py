import multiprocessing
import subprocess
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_video_processing(gpu_id, video_folder_name, base_input_path, base_output_path, base_csv_path):
    """Run video processing for a specific video folder on a specific GPU"""
    
    # Construct paths
    input_folder = os.path.join(base_input_path, video_folder_name, "video")
    output_folder = os.path.join(base_output_path, video_folder_name)
    csv_folder = os.path.join(base_csv_path, video_folder_name)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        return False, video_folder_name, gpu_id
    
    print(f"🚀 Starting {video_folder_name} on GPU {gpu_id}")
    
    # Construct command
    cmd = [
        sys.executable, "frame_extraction_gpu.py",
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--csv_folder", csv_folder,
        "--gpu_id", str(gpu_id)
    ]
    
    try:
        # Run the process
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {video_folder_name} completed successfully on GPU {gpu_id} (Duration: {duration:.2f}s)")
            print(f"   Output: {result.stdout}")
            return True, video_folder_name, gpu_id
        else:
            print(f"❌ {video_folder_name} failed on GPU {gpu_id}")
            print(f"   Error: {result.stderr}")
            return False, video_folder_name, gpu_id
            
    except Exception as e:
        print(f"❌ Exception running {video_folder_name} on GPU {gpu_id}: {e}")
        return False, video_folder_name, gpu_id

def run_parallel_processing(video_folders, base_input_path, base_output_path, base_csv_path, num_gpus=10):
    """Run video processing in parallel across multiple GPUs"""
    
    print(f"🎯 Starting parallel processing with {num_gpus} GPUs")
    print(f"📁 Processing folders: {video_folders}")
    print(f"📥 Input base path: {base_input_path}")
    print(f"📤 Output base path: {base_output_path}")
    print(f"📊 CSV base path: {base_csv_path}")
    print("-" * 60)
    
    # Create tasks list (video_folder, gpu_id)
    tasks = []
    for i, video_folder in enumerate(video_folders):
        gpu_id = i % num_gpus  # Distribute folders across available GPUs
        tasks.append((gpu_id, video_folder, base_input_path, base_output_path, base_csv_path))
    
    # Track results
    completed_tasks = []
    failed_tasks = []
    
    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_video_processing, *task): task 
            for task in tasks
        }
        
        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            gpu_id, video_folder = task[0], task[1]
            
            try:
                success, folder_name, gpu_used = future.result()
                if success:
                    completed_tasks.append((folder_name, gpu_used))
                else:
                    failed_tasks.append((folder_name, gpu_used))
            except Exception as e:
                print(f"❌ Task {video_folder} on GPU {gpu_id} generated an exception: {e}")
                failed_tasks.append((video_folder, gpu_id))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"✅ Completed: {len(completed_tasks)} folders")
    for folder, gpu in completed_tasks:
        print(f"   - {folder} (GPU {gpu})")
    
    print(f"\n❌ Failed: {len(failed_tasks)} folders")
    for folder, gpu in failed_tasks:
        print(f"   - {folder} (GPU {gpu})")
    
    print(f"\n🎯 Success rate: {len(completed_tasks)}/{len(video_folders)} ({len(completed_tasks)/len(video_folders)*100:.1f}%)")
    
    return len(failed_tasks) == 0

def main():
    # Configuration
    VIDEO_FOLDERS = [f"Videos_K0{i}" for i in range(1, 5)]  # Videos_K1 to Videos_K4
    BASE_INPUT_PATH = "/home/tuktu/KeyFrames_Extraction_Server/data"  # Update this path
    BASE_OUTPUT_PATH = "/home/tuktu/KeyFrames_Extraction_Server/output/images"  # Update this path
    BASE_CSV_PATH = "/home/tuktu/KeyFrames_Extraction_Server/output/csv"  # Update this path
    NUM_GPUS = 4
    
    # Display configuration
    print("🔧 CONFIGURATION")
    print("=" * 60)
    print(f"Video folders to process: {VIDEO_FOLDERS}")
    print(f"Base input path: {BASE_INPUT_PATH}")
    print(f"Base output path: {BASE_OUTPUT_PATH}")
    print(f"Base CSV path: {BASE_CSV_PATH}")
    print(f"Number of GPUs: {NUM_GPUS}")
    print("=" * 60)
    
    # Validate paths
    if not os.path.exists(BASE_INPUT_PATH):
        print(f"❌ Base input path does not exist: {BASE_INPUT_PATH}")
        print("Please update BASE_INPUT_PATH in the script")
        return
    
    # Create output directories
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    os.makedirs(BASE_CSV_PATH, exist_ok=True)
    
    # Start processing
    start_time = time.time()
    success = run_parallel_processing(
        video_folders=VIDEO_FOLDERS,
        base_input_path=BASE_INPUT_PATH,
        base_output_path=BASE_OUTPUT_PATH,
        base_csv_path=BASE_CSV_PATH,
        num_gpus=NUM_GPUS
    )
    end_time = time.time()
    
    total_duration = end_time - start_time
    print(f"\n⏱️  Total processing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    if success:
        print("🎉 All video folders processed successfully!")
    else:
        print("⚠️  Some video folders failed to process. Check the logs above.")

if __name__ == "__main__":
    main()
