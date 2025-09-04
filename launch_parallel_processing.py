#!/usr/bin/env python3
"""
Simple launcher script for parallel GPU processing
Update the paths in the configuration section below
"""

import os
import sys

def main():
    # ========== CONFIGURATION - UPDATE THESE PATHS ==========
    BASE_INPUT_PATH = r"D:\path\to\your\input\data"  # Update this path
    BASE_OUTPUT_PATH = r"D:\path\to\your\output\images"  # Update this path  
    BASE_CSV_PATH = r"D:\path\to\your\output\csv"  # Update this path
    
    # ========== DO NOT MODIFY BELOW THIS LINE ==========
    
    print("🔧 PARALLEL GPU PROCESSING LAUNCHER")
    print("=" * 60)
    print(f"Base input path: {BASE_INPUT_PATH}")
    print(f"Base output path: {BASE_OUTPUT_PATH}")
    print(f"Base CSV path: {BASE_CSV_PATH}")
    print(f"Processing: Videos_K10 to Videos_K20")
    print(f"GPUs: 0-9 (10 GPUs total)")
    print("=" * 60)
    
    # Validate input path
    if not os.path.exists(BASE_INPUT_PATH):
        print(f"❌ Base input path does not exist: {BASE_INPUT_PATH}")
        print("Please update BASE_INPUT_PATH in this script")
        input("Press Enter to exit...")
        return
    
    # Create output directories
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    os.makedirs(BASE_CSV_PATH, exist_ok=True)
    print(f"📁 Output directories ensured")
    
    # Ask for confirmation
    print(f"\n⚠️  Ready to start parallel processing?")
    print(f"This will process 11 video folders (Videos_K10 to Videos_K20) across 10 GPUs")
    confirmation = input("Continue? (y/N): ").strip().lower()
    
    if confirmation != 'y':
        print("❌ Processing cancelled")
        return
    
    # Update the paths in parallel_gpu_processing.py
    try:
        # Read the original file
        with open('parallel_gpu_processing.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace the placeholder paths
        content = content.replace('BASE_INPUT_PATH = "/path/to/your/input/data"', 
                                f'BASE_INPUT_PATH = r"{BASE_INPUT_PATH}"')
        content = content.replace('BASE_OUTPUT_PATH = "/path/to/your/output/images"', 
                                f'BASE_OUTPUT_PATH = r"{BASE_OUTPUT_PATH}"')
        content = content.replace('BASE_CSV_PATH = "/path/to/your/output/csv"', 
                                f'BASE_CSV_PATH = r"{BASE_CSV_PATH}"')
        
        # Write the updated content to a temporary file
        with open('temp_parallel_processing.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("\n🚀 Starting parallel GPU processing...")
        print("=" * 60)
        
        # Run the processing
        import subprocess
        result = subprocess.run([sys.executable, 'temp_parallel_processing.py'], 
                              capture_output=False, text=True)
        
        # Clean up
        if os.path.exists('temp_parallel_processing.py'):
            os.remove('temp_parallel_processing.py')
        
        if result.returncode == 0:
            print("\n🎉 Processing completed successfully!")
        else:
            print(f"\n❌ Processing failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        if os.path.exists('temp_parallel_processing.py'):
            os.remove('temp_parallel_processing.py')
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
