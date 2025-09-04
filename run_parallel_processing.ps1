# PowerShell script to run parallel GPU processing
# Update these paths according to your setup

# Configuration
$BASE_INPUT_PATH = "D:\path\to\your\input\data"  # Update this path
$BASE_OUTPUT_PATH = "D:\path\to\your\output\images"  # Update this path  
$BASE_CSV_PATH = "D:\path\to\your\output\csv"  # Update this path

# Display configuration
Write-Host "🔧 PARALLEL GPU PROCESSING CONFIGURATION" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray
Write-Host "Base input path: $BASE_INPUT_PATH" -ForegroundColor White
Write-Host "Base output path: $BASE_OUTPUT_PATH" -ForegroundColor White
Write-Host "Base CSV path: $BASE_CSV_PATH" -ForegroundColor White
Write-Host "Processing: Videos_K10 to Videos_K20" -ForegroundColor White
Write-Host "GPUs: 0-9 (10 GPUs total)" -ForegroundColor White
Write-Host "=" * 60 -ForegroundColor Gray

# Check if input path exists
if (-not (Test-Path $BASE_INPUT_PATH)) {
    Write-Host "❌ Base input path does not exist: $BASE_INPUT_PATH" -ForegroundColor Red
    Write-Host "Please update the BASE_INPUT_PATH in this script" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create output directories if they don't exist
if (-not (Test-Path $BASE_OUTPUT_PATH)) {
    New-Item -Path $BASE_OUTPUT_PATH -ItemType Directory -Force
    Write-Host "📁 Created output directory: $BASE_OUTPUT_PATH" -ForegroundColor Green
}

if (-not (Test-Path $BASE_CSV_PATH)) {
    New-Item -Path $BASE_CSV_PATH -ItemType Directory -Force
    Write-Host "📁 Created CSV directory: $BASE_CSV_PATH" -ForegroundColor Green
}

# Ask for confirmation
Write-Host "`n⚠️  Ready to start parallel processing?" -ForegroundColor Yellow
Write-Host "This will process 11 video folders (Videos_K10 to Videos_K20) across 10 GPUs" -ForegroundColor White
$confirmation = Read-Host "Continue? (y/N)"

if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "❌ Processing cancelled" -ForegroundColor Red
    exit 0
}

# Start processing
Write-Host "`n🚀 Starting parallel GPU processing..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Gray

try {
    # Update the parallel_gpu_processing.py script with the correct paths
    $pythonScript = @"
import multiprocessing
import subprocess
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_video_processing(gpu_id, video_folder_name, base_input_path, base_output_path, base_csv_path):
    input_folder = os.path.join(base_input_path, video_folder_name, "video")
    output_folder = os.path.join(base_output_path, video_folder_name)
    csv_folder = os.path.join(base_csv_path, video_folder_name)
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        return False, video_folder_name, gpu_id
    
    print(f"🚀 Starting {video_folder_name} on GPU {gpu_id}")
    
    cmd = [
        sys.executable, "frame_extraction_gpu.py",
        "--input_folder", input_folder,
        "--output_folder", output_folder,
        "--csv_folder", csv_folder,
        "--gpu_id", str(gpu_id)
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {video_folder_name} completed successfully on GPU {gpu_id} (Duration: {duration:.2f}s)")
            return True, video_folder_name, gpu_id
        else:
            print(f"❌ {video_folder_name} failed on GPU {gpu_id}")
            print(f"   Error: {result.stderr}")
            return False, video_folder_name, gpu_id
            
    except Exception as e:
        print(f"❌ Exception running {video_folder_name} on GPU {gpu_id}: {e}")
        return False, video_folder_name, gpu_id

# Configuration
VIDEO_FOLDERS = [f"Videos_K{i}" for i in range(10, 21)]
BASE_INPUT_PATH = r"$BASE_INPUT_PATH"
BASE_OUTPUT_PATH = r"$BASE_OUTPUT_PATH"
BASE_CSV_PATH = r"$BASE_CSV_PATH"
NUM_GPUS = 10

tasks = []
for i, video_folder in enumerate(VIDEO_FOLDERS):
    gpu_id = i % NUM_GPUS
    tasks.append((gpu_id, video_folder, BASE_INPUT_PATH, BASE_OUTPUT_PATH, BASE_CSV_PATH))

completed_tasks = []
failed_tasks = []

print(f"🎯 Starting parallel processing with {NUM_GPUS} GPUs")
print(f"📁 Processing folders: {VIDEO_FOLDERS}")

with ProcessPoolExecutor(max_workers=NUM_GPUS) as executor:
    future_to_task = {executor.submit(run_video_processing, *task): task for task in tasks}
    
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

print("\n" + "=" * 60)
print("📊 PROCESSING SUMMARY")
print("=" * 60)
print(f"✅ Completed: {len(completed_tasks)} folders")
for folder, gpu in completed_tasks:
    print(f"   - {folder} (GPU {gpu})")

print(f"\n❌ Failed: {len(failed_tasks)} folders")
for folder, gpu in failed_tasks:
    print(f"   - {folder} (GPU {gpu})")

print(f"\n🎯 Success rate: {len(completed_tasks)}/{len(VIDEO_FOLDERS)} ({len(completed_tasks)/len(VIDEO_FOLDERS)*100:.1f}%)")
"@

    # Write the temporary Python script
    $pythonScript | Out-File -FilePath "temp_parallel_processing.py" -Encoding UTF8
    
    # Run the Python script
    python temp_parallel_processing.py
    
    # Clean up temporary file
    Remove-Item "temp_parallel_processing.py" -ErrorAction SilentlyContinue
    
    Write-Host "`n🎉 Processing completed!" -ForegroundColor Green
    
} catch {
    Write-Host "`n❌ Error occurred during processing:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
} finally {
    Write-Host "`nPress Enter to exit..." -ForegroundColor Gray
    Read-Host
}
