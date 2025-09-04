#!/usr/bin/env python3
"""
GPU Test Script for KeyFrames Extraction Server
This script tests if GPUs are properly configured and can load the TransNetV2 model
"""

import os
import sys
sys.path.append("/home/tuktu/KeyFrames_Extraction_Server/TransNetV2/inference")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from transnetv2 import TransNetV2
import numpy as np

def test_gpu_basic():
    """Test basic GPU functionality"""
    print("🔍 Testing basic GPU functionality...")
    
    try:
        # Check if CUDA is available
        print(f"   CUDA built: {tf.test.is_built_with_cuda()}")
        print(f"   GPU available: {tf.test.is_gpu_available()}")
        
        # List physical GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"   Physical GPUs found: {len(gpus)}")
        
        if not gpus:
            print("❌ No GPUs found!")
            return False
            
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
            
        return True
        
    except Exception as e:
        print(f"❌ Basic GPU test failed: {e}")
        return False

def test_gpu_operation(gpu_id):
    """Test GPU operations on specific GPU"""
    print(f"🧪 Testing GPU {gpu_id} operations...")
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpu_id >= len(gpus):
            print(f"❌ GPU {gpu_id} not available (found {len(gpus)} GPUs)")
            return False
            
        # Configure GPU
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        
        # Test tensor operations
        with tf.device(f'/GPU:{gpu_id}'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[2.0, 0.0], [0.0, 2.0]])
            result = tf.matmul(a, b)
            
        print(f"✅ GPU {gpu_id} tensor operations successful")
        print(f"   Test result shape: {result.shape}")
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} operation test failed: {e}")
        return False

def test_transnetv2_model(gpu_id):
    """Test loading TransNetV2 model on specific GPU"""
    print(f"🤖 Testing TransNetV2 model loading on GPU {gpu_id}...")
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpu_id >= len(gpus):
            print(f"❌ GPU {gpu_id} not available")
            return False
            
        # Configure GPU
        tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        
        # Load model
        with tf.device(f'/GPU:{gpu_id}'):
            model = TransNetV2()
            
        print(f"✅ TransNetV2 model loaded successfully on GPU {gpu_id}")
        
        # Test with dummy data
        dummy_frames = np.random.rand(10, 48, 27, 3).astype(np.float32)
        
        with tf.device(f'/GPU:{gpu_id}'):
            predictions, scene_changes = model.predict_frames(dummy_frames)
            
        print(f"✅ Model prediction test successful on GPU {gpu_id}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Scene changes shape: {scene_changes.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ TransNetV2 model test failed on GPU {gpu_id}: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting GPU Test Suite for KeyFrames Extraction Server")
    print("=" * 60)
    
    # Test 1: Basic GPU functionality
    if not test_gpu_basic():
        print("❌ Basic GPU test failed. Exiting.")
        return False
    
    print("\n" + "-" * 40)
    
    # Test 2: Test each GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    successful_gpus = []
    failed_gpus = []
    
    for gpu_id in range(min(num_gpus, 4)):  # Test first 4 GPUs
        print(f"\n🔧 Testing GPU {gpu_id}:")
        
        # Test GPU operations
        if test_gpu_operation(gpu_id):
            # Test TransNetV2 model
            if test_transnetv2_model(gpu_id):
                successful_gpus.append(gpu_id)
                print(f"✅ GPU {gpu_id} passed all tests")
            else:
                failed_gpus.append(gpu_id)
                print(f"❌ GPU {gpu_id} failed model test")
        else:
            failed_gpus.append(gpu_id)
            print(f"❌ GPU {gpu_id} failed operation test")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"✅ Successful GPUs: {successful_gpus}")
    print(f"❌ Failed GPUs: {failed_gpus}")
    
    if successful_gpus:
        print(f"🎉 {len(successful_gpus)} GPU(s) ready for parallel processing!")
        print(f"💡 You can use GPUs: {successful_gpus}")
        return True
    else:
        print("❌ No GPUs passed all tests. Check your CUDA/GPU setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
