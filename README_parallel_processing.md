# Parallel GPU Video Processing

This system allows you to process multiple video folders (Videos_K10 to Videos_K20) in parallel across multiple GPUs for keyframe extraction.

## Files Created

1. **`frame_extraction_gpu.py`** - Modified version of your original script that accepts GPU ID parameter
2. **`parallel_gpu_processing.py`** - Main parallel processing script
3. **`launch_parallel_processing.py`** - Simple Python launcher (recommended)
4. **`run_parallel_processing.ps1`** - PowerShell launcher for Windows

## Quick Setup

### Step 1: Update Paths
Edit the paths in `launch_parallel_processing.py`:

```python
BASE_INPUT_PATH = r"D:\path\to\your\input\data"      # Where Videos_K10, Videos_K11, etc. are located
BASE_OUTPUT_PATH = r"D:\path\to\your\output\images"  # Where extracted frames will be saved
BASE_CSV_PATH = r"D:\path\to\your\output\csv"        # Where CSV files will be saved
```

### Step 2: Run the Processing
```bash
python launch_parallel_processing.py
```

## How It Works

### GPU Distribution
- Videos_K10 → GPU 0
- Videos_K11 → GPU 1
- Videos_K12 → GPU 2
- ...
- Videos_K19 → GPU 9
- Videos_K20 → GPU 0 (cycles back)

### Folder Structure Expected
```
BASE_INPUT_PATH/
├── Videos_K10/
│   └── video/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── ...
├── Videos_K11/
│   └── video/
│       ├── video1.mp4
│       └── ...
└── ...
```

### Output Structure
```
BASE_OUTPUT_PATH/
├── Videos_K10/
│   ├── video1/
│   │   ├── 000.jpg
│   │   ├── 001.jpg
│   │   └── ...
│   └── video2/
│       └── ...
└── ...

BASE_CSV_PATH/
├── Videos_K10/
│   ├── video1.csv
│   ├── video2.csv
│   └── ...
└── ...
```

## Manual Usage

If you prefer to run manually or customize further:

### Single GPU Processing
```bash
python frame_extraction_gpu.py --input_folder "D:\data\Videos_K10\video" --output_folder "D:\output\Videos_K10" --csv_folder "D:\csv\Videos_K10" --gpu_id 0
```

### Custom Parallel Processing
Edit `parallel_gpu_processing.py` and update the configuration section, then run:
```bash
python parallel_gpu_processing.py
```

## Monitoring

The system provides real-time status updates:
- 🚀 Starting processing for each folder
- ✅ Successful completion with duration
- ❌ Error messages for failed processing
- 📊 Final summary with success rate

## GPU Memory Management

The system automatically:
- Sets visible devices to specific GPU
- Enables memory growth to prevent OOM errors
- Loads TransNetV2 model after GPU configuration

## Troubleshooting

### Common Issues

1. **GPU not found**: Check if CUDA and GPUs are properly configured
2. **Input folder not found**: Verify the folder structure matches expectations
3. **Memory errors**: Reduce batch size or enable memory growth (already implemented)
4. **Permission errors**: Ensure write permissions for output directories

### Checking GPU Availability
```python
import tensorflow as tf
print("GPUs Available:", tf.config.experimental.list_physical_devices('GPU'))
```

### Manual GPU Test
```bash
python frame_extraction_gpu.py --input_folder "test_folder" --output_folder "test_output" --csv_folder "test_csv" --gpu_id 0
```

## Performance Tips

1. **SSD Storage**: Use SSD for input/output for better I/O performance
2. **RAM**: Ensure sufficient RAM for multiple processes
3. **GPU Memory**: Monitor GPU memory usage with `nvidia-smi`
4. **Cooling**: Ensure adequate cooling for sustained GPU usage

## Customization

### Adjusting Number of GPUs
Edit `NUM_GPUS` in the script files to match your setup.

### Changing Video Folder Range
Modify the `VIDEO_FOLDERS` list in `parallel_gpu_processing.py`:
```python
VIDEO_FOLDERS = [f"Videos_K{i}" for i in range(10, 21)]  # K10 to K20
# Or specify manually:
VIDEO_FOLDERS = ["Videos_K15", "Videos_K16", "Videos_K17"]
```

### Processing Parameters
Modify the constants in `frame_extraction_gpu.py`:
```python
HAMMING_THRESHOLD = 5          # Duplicate frame threshold
KEYFRAME_PROB_THRESHOLD = 0.2  # Scene change threshold
```
