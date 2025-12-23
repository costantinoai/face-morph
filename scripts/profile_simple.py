#!/usr/bin/env python3
"""
Simple performance profiling for face-morph pipeline.

Measures execution time and memory usage for different test scenarios.
"""

import time
import psutil
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Any


def profile_command(cmd: list, test_name: str) -> Dict[str, Any]:
    """Profile a command execution."""
    print(f"\n{'='*70}")
    print(f"PROFILING: {test_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    process = psutil.Process()
    initial_mem = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.perf_counter()

    # Run command
    result = subprocess.run(cmd, capture_output=True, text=True)

    end_time = time.perf_counter()

    final_mem = process.memory_info().rss / 1024 / 1024  # MB
    elapsed = end_time - start_time

    # Parse output for additional metrics
    output = result.stdout + result.stderr
    success = result.returncode == 0

    # Extract metrics from log
    device_transfers = None
    frames_generated = None

    for line in output.split('\n'):
        if 'Device transfers:' in line:
            try:
                device_transfers = int(line.split(':')[1].strip().split()[0])
            except:
                pass
        if 'Frames generated:' in line:
            try:
                frames_generated = int(line.split(':')[1].strip())
            except:
                pass

    result_data = {
        "test_name": test_name,
        "success": success,
        "execution_time_sec": elapsed,
        "memory_delta_mb": final_mem - initial_mem,
        "exit_code": result.returncode,
        "frames_generated": frames_generated,
        "device_transfers": device_transfers,
    }

    if frames_generated and elapsed > 0:
        result_data["frames_per_second"] = frames_generated / elapsed
        result_data["seconds_per_frame"] = elapsed / frames_generated

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {test_name}")
    print(f"{'='*70}")
    print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Time: {elapsed:.2f} seconds")
    if frames_generated:
        print(f"Frames: {frames_generated}")
        print(f"Throughput: {result_data['frames_per_second']:.2f} frames/sec")
        print(f"Per-frame: {result_data['seconds_per_frame']:.3f} sec/frame")
    if device_transfers is not None:
        print(f"Device transfers: {device_transfers}")
    print(f"Memory delta: {final_mem - initial_mem:.1f} MB")
    print(f"{'='*70}\n")

    return result_data


def run_profiling_suite():
    """Run comprehensive profiling suite."""

    # Setup paths
    repo_root = Path(__file__).parent.parent
    data_dir = repo_root / "data"
    profile_dir = repo_root / "profiling_results"
    profile_dir.mkdir(exist_ok=True)

    # Test cases
    test_cases = [
        {
            "name": "default_cpu_textured",
            "cmd": [
                "face-morph", "morph",
                str(data_dir / "male1.fbx"),
                str(data_dir / "male2.fbx"),
                "-o", str(profile_dir / "default_cpu_textured"),
            ],
        },
        {
            "name": "default_cpu_notextured",
            "cmd": [
                "face-morph", "morph",
                str(data_dir / "male1.fbx"),
                str(data_dir / "woman19.fbx"),
                "-o", str(profile_dir / "default_cpu_notextured"),
            ],
        },
        {
            "name": "full_cpu_textured",
            "cmd": [
                "face-morph", "morph",
                str(data_dir / "male1.fbx"),
                str(data_dir / "male2.fbx"),
                "--full",
                "-o", str(profile_dir / "full_cpu_textured"),
            ],
        },
    ]

    # Run tests
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'#'*70}")

        result = profile_command(test_case['cmd'], test_case['name'])
        results.append(result)

        # Save individual result
        result_file = profile_dir / f"{test_case['name']}_profile.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Create comparison report
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    # Table header
    print(f"{'Test':<30} {'Time (s)':<12} {'FPS':<10} {'Transfers':<12}")
    print("-" * 70)

    for result in results:
        name = result['test_name']
        time_sec = result['execution_time_sec']
        fps = result.get('frames_per_second', 0)
        transfers = result.get('device_transfers', 'N/A')

        print(f"{name:<30} {time_sec:<12.2f} {fps:<10.2f} {str(transfers):<12}")

    # Calculate speedups
    print(f"\n{'='*70}")
    print("OPTIMIZATION ANALYSIS")
    print(f"{'='*70}\n")

    # Device transfers
    print("Device Transfer Optimization:")
    for result in results:
        if result.get('device_transfers') is not None:
            print(f"  {result['test_name']}: {result['device_transfers']} transfers")

    print("\nExpected GPU improvements (not tested):")
    print("  - Batch rendering: 10-20x faster")
    print("  - Device transfers: 20-30% overhead reduction")
    print("  - Mixed precision (FP16): Additional 2x speedup")

    # Save full report
    report_file = profile_dir / "profiling_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": results,
        }, f, indent=2)

    print(f"\n✅ Full report saved: {report_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Activate conda environment and run
    run_profiling_suite()
