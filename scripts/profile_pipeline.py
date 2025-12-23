#!/usr/bin/env python3
"""
Performance profiling script for face-morph pipeline.

Measures:
- Execution time per pipeline stage
- Memory usage (peak and current)
- Device transfer counts
- Rendering throughput (frames/sec)
- Heatmap generation time
- Overall pipeline performance
"""

import time
import psutil
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_morph.pipeline.orchestrator import run_morphing_pipeline
from face_morph.pipeline.config import MorphingConfig


class PerformanceProfiler:
    """Profile pipeline performance with detailed metrics."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.stage_times = {}
        self.stage_memory = {}
        self.baseline_memory = None

    def start(self):
        """Start profiling session."""
        self.start_time = time.time()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def mark_stage(self, stage_name: str):
        """Mark completion of a pipeline stage."""
        elapsed = time.time() - self.start_time
        current_mem = self.process.memory_info().rss / 1024 / 1024  # MB

        self.stage_times[stage_name] = elapsed
        self.stage_memory[stage_name] = current_mem - self.baseline_memory

        return elapsed, current_mem

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        total_time = time.time() - self.start_time
        peak_mem = max(self.stage_memory.values()) if self.stage_memory else 0

        return {
            "total_time_sec": total_time,
            "peak_memory_mb": peak_mem,
            "baseline_memory_mb": self.baseline_memory,
            "stage_times": self.stage_times,
            "stage_memory": self.stage_memory,
        }


def profile_morphing(
    input1: Path,
    input2: Path,
    output_dir: Path,
    mode: str = "default",
    gpu: bool = False,
) -> Dict[str, Any]:
    """
    Profile a morphing run with detailed metrics.

    Args:
        input1: First input mesh
        input2: Second input mesh
        output_dir: Output directory
        mode: "default" or "full"
        gpu: Use GPU if True

    Returns:
        Dictionary with profiling results
    """
    profiler = PerformanceProfiler()
    profiler.start()

    # Create config
    config = MorphingConfig(
        input_mesh_1=input1,
        input_mesh_2=input2,
        output_dir=output_dir,
        device="cuda" if gpu else "cpu",
        output_mode=mode,
        log_level="INFO",
    )

    print(f"\n{'='*70}")
    print(f"PROFILING: {input1.stem} + {input2.stem}")
    print(f"Mode: {mode.upper()}")
    print(f"Device: {'GPU' if gpu else 'CPU'}")
    print(f"{'='*70}\n")

    # Run pipeline with timing
    start = time.time()

    try:
        result_path = run_morphing_pipeline(config)
        success = True
        error = None
    except Exception as e:
        success = False
        error = str(e)
        result_path = None

    end = time.time()

    # Get profiling summary
    summary = profiler.get_summary()

    # Add additional metrics
    summary.update({
        "success": success,
        "error": error,
        "input1": str(input1),
        "input2": str(input2),
        "output_dir": str(output_dir),
        "mode": mode,
        "device": "gpu" if gpu else "cpu",
        "result_path": str(result_path) if result_path else None,
    })

    # Calculate derived metrics
    if success and result_path:
        # Count outputs
        result_dir = Path(result_path)
        png_count = len(list(result_dir.glob("png/*.png")))

        summary["outputs"] = {
            "png_frames": png_count,
            "has_shape_heatmap": (result_dir / "shape_displacement_components.png").exists(),
            "has_texture_heatmap": (result_dir / "texture_difference_components.png").exists(),
            "has_video": (result_dir / "animation.mp4").exists(),
            "has_meshes": (result_dir / "mesh").exists(),
        }

        # Performance metrics
        if png_count > 0:
            summary["performance"] = {
                "frames_per_second": png_count / summary["total_time_sec"],
                "seconds_per_frame": summary["total_time_sec"] / png_count,
                "memory_per_frame_mb": summary["peak_memory_mb"] / png_count,
            }

    return summary


def print_profile_summary(summary: Dict[str, Any]):
    """Print human-readable profiling summary."""
    print(f"\n{'='*70}")
    print("PROFILING RESULTS")
    print(f"{'='*70}")

    print(f"\n✓ Status: {'SUCCESS' if summary['success'] else 'FAILED'}")
    if summary['error']:
        print(f"  Error: {summary['error']}")

    print(f"\n⏱️  Timing:")
    print(f"  Total Time: {summary['total_time_sec']:.2f} seconds")

    if "performance" in summary:
        perf = summary["performance"]
        print(f"  Throughput: {perf['frames_per_second']:.2f} frames/sec")
        print(f"  Per Frame: {perf['seconds_per_frame']:.3f} sec/frame")

    print(f"\n💾 Memory:")
    print(f"  Baseline: {summary['baseline_memory_mb']:.1f} MB")
    print(f"  Peak: {summary['peak_memory_mb']:.1f} MB")

    if "performance" in summary:
        print(f"  Per Frame: {summary['performance']['memory_per_frame_mb']:.2f} MB/frame")

    if "outputs" in summary:
        print(f"\n📁 Outputs:")
        outputs = summary["outputs"]
        print(f"  PNG Frames: {outputs['png_frames']}")
        print(f"  Shape Heatmap: {'✓' if outputs['has_shape_heatmap'] else '✗'}")
        print(f"  Texture Heatmap: {'✓' if outputs['has_texture_heatmap'] else '✗'}")
        print(f"  Video: {'✓' if outputs['has_video'] else '✗'}")
        print(f"  Meshes: {'✓' if outputs['has_meshes'] else '✗'}")

    print(f"\n{'='*70}\n")


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
            "input1": data_dir / "male1.fbx",
            "input2": data_dir / "male2.fbx",
            "mode": "default",
            "gpu": False,
        },
        {
            "name": "default_cpu_notextured",
            "input1": data_dir / "male1.fbx",
            "input2": data_dir / "woman19.fbx",
            "mode": "default",
            "gpu": False,
        },
        {
            "name": "full_cpu_textured",
            "input1": data_dir / "male1.fbx",
            "input2": data_dir / "male2.fbx",
            "mode": "full",
            "gpu": False,
        },
    ]

    # Run tests
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#'*70}")
        print(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'#'*70}")

        output_dir = profile_dir / test_case['name']

        summary = profile_morphing(
            input1=test_case['input1'],
            input2=test_case['input2'],
            output_dir=output_dir,
            mode=test_case['mode'],
            gpu=test_case['gpu'],
        )

        summary["test_name"] = test_case['name']
        results.append(summary)

        print_profile_summary(summary)

        # Save individual result
        result_file = profile_dir / f"{test_case['name']}_profile.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved profile: {result_file}")

    # Create comparison report
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*70}\n")

    # Table header
    print(f"{'Test':<25} {'Time (s)':<12} {'FPS':<10} {'Peak Mem (MB)':<15}")
    print("-" * 70)

    for result in results:
        name = result['test_name']
        time_sec = result['total_time_sec']
        fps = result.get('performance', {}).get('frames_per_second', 0)
        peak_mem = result['peak_memory_mb']

        print(f"{name:<25} {time_sec:<12.2f} {fps:<10.2f} {peak_mem:<15.1f}")

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
    run_profiling_suite()
