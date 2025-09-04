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
    
    # Construct command with environment variables for GPU isolation
    cmd = [
        sys.executable, "frame_extraction_gpu_isolated_clean.py",
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--csv_folder", csv_folder,
        "--gpu_id", str(gpu_id)
    ]
    
    # Set environment variables to isolate GPU usage
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    try:
        # Run the process
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {video_folder_name} completed successfully on GPU {gpu_id} (Duration: {duration:.2f}s)")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True, video_folder_name, gpu_id
        else:
            print(f"❌ {video_folder_name} failed on GPU {gpu_id}")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False, video_folder_name, gpu_id
            
    except Exception as e:
        print(f"❌ Exception running {video_folder_name} on GPU {gpu_id}: {e}")
        return False, video_folder_name, gpu_id

def run_parallel_processing(video_folders, base_input_path, base_output_path, base_csv_path, num_gpus=4):
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
    print("📊 PROCESSING SUMMARY:")
    print(f"✅ Completed tasks: {len(completed_tasks)}")
    print(f"❌ Failed tasks: {len(failed_tasks)}")
    
    if completed_tasks:
        print("\n✅ Successfully processed:")
        for folder, gpu in completed_tasks:
            print(f"   {folder} (GPU {gpu})")
    
    if failed_tasks:
        print("\n❌ Failed to process:")
        for folder, gpu in failed_tasks:
            print(f"   {folder} (GPU {gpu})")
    
    return len(completed_tasks), len(failed_tasks)

def main():
    """Main function for running parallel GPU processing"""
    
    # Configuration
    video_folders = ['Videos_K01', 'Videos_K02', 'Videos_K03', 'Videos_K04']
    base_input_path = "/home/tuktu/KeyFrames_Extraction_Server/data"
    base_output_path = "/home/tuktu/KeyFrames_Extraction_Server/output/images"
    base_csv_path = "/home/tuktu/KeyFrames_Extraction_Server/output/csv"
    num_gpus = 4
    
    # Run parallel processing
    completed, failed = run_parallel_processing(
        video_folders, 
        base_input_path, 
        base_output_path, 
        base_csv_path, 
        num_gpus
    )
    
    # Exit with appropriate code
    if failed == 0:
        print("🎉 All tasks completed successfully!")
        sys.exit(0)
    else:
        print(f"⚠️  {failed} tasks failed out of {completed + failed}")
        sys.exit(1)

if __name__ == "__main__":
    main()
