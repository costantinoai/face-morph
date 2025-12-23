"""
Command-Line Interface
======================

Single responsibility: Provide user-friendly CLI for face morphing.
"""

import click
from pathlib import Path
import sys
import torch
from itertools import combinations

from face_morph.pipeline import MorphConfig, run_morphing_pipeline
from face_morph.utils.logging import get_logger

logger = get_logger(__name__)


@click.command()
@click.argument('input1', type=click.Path(exists=True, path_type=Path))
@click.argument('input2', type=click.Path(exists=True, path_type=Path))
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory (default: results/)'
)
@click.option(
    '--full',
    is_flag=True,
    help='Generate full output: PNG + heatmaps + meshes + video + CSV (~2 hours)'
)
@click.option(
    '--gpu/--cpu',
    default=False,
    help='Use GPU acceleration (default: CPU)'
)
@click.option(
    '--no-amp',
    is_flag=True,
    help='Disable mixed precision (FP16) on GPU'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    default=True,
    help='Verbose output (default: enabled)'
)
@click.option(
    '-q', '--quiet',
    is_flag=True,
    help='Suppress output'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
    default='INFO',
    help='Logging level (default: INFO)'
)
@click.option(
    '--blender',
    type=str,
    default='blender',
    help='Path to Blender executable (default: blender)'
)
def morph(input1, input2, output, full, gpu, no_amp, verbose, quiet, log_level, blender):
    """
    Morph two 3D face meshes with interpolation.

    By default, generates PNG images and heatmaps only (fast mode, ~5-10 min).
    Use --full to also export meshes, create video, and export CSV data (complete mode, ~2 hours).

    \b
    Examples:
        # Fast mode (PNG + heatmaps only)
        face-morph face1.fbx face2.fbx

        # Full mode (PNG + heatmaps + meshes + video + CSV)
        face-morph face1.fbx face2.fbx --full --gpu

        # Custom output directory
        face-morph face1.obj face2.obj -o custom_results/

        # CPU mode with verbose debug logging
        face-morph face1.fbx face2.fbx --cpu --log-level DEBUG
    """
    # Handle verbose/quiet flags
    if quiet:
        verbose = False

    # Create config
    try:
        config = MorphConfig(
            input_mesh_1=input1,
            input_mesh_2=input2,
            output_dir=output or Path('results'),
            output_mode='full' if full else 'default',
            device=torch.device('cuda' if gpu else 'cpu'),
            use_mixed_precision=not no_amp,
            blender_path=blender,
            verbose=verbose,
            log_level=log_level.upper()
        )
    except Exception as e:
        click.secho(f"\n✗ Configuration error: {e}", fg='red', bold=True)
        sys.exit(1)

    # Run pipeline
    try:
        click.echo()
        output_path = run_morphing_pipeline(config)
        click.echo()
        click.secho(f"✓ Success! Results saved to: {output_path}", fg='green', bold=True)
        click.echo()
    except KeyboardInterrupt:
        click.echo()
        click.secho("\n✗ Interrupted by user", fg='yellow')
        sys.exit(130)
    except Exception as e:
        click.echo()
        click.secho(f"✗ Error: {e}", fg='red', bold=True)
        if verbose or log_level.upper() == 'DEBUG':
            import traceback
            click.echo()
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.argument('folder', type=click.Path(exists=True, path_type=Path))
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    default=None,
    help='Output directory (default: results/)'
)
@click.option(
    '--full',
    is_flag=True,
    help='Generate full output for each pair'
)
@click.option(
    '--gpu/--cpu',
    default=False,
    help='Use GPU acceleration (default: CPU)'
)
@click.option(
    '-v', '--verbose',
    is_flag=True,
    default=True,
    help='Verbose output'
)
@click.option(
    '--log-level',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR'], case_sensitive=False),
    default='INFO',
    help='Logging level (default: INFO)'
)
def batch(folder, output, full, gpu, verbose, log_level):
    """
    Batch process all unique pairs in a folder.

    Discovers all .fbx and .obj files, generates all unique morphing combinations
    (excludes self-pairs, treats pairs as unordered).

    \b
    Examples:
        # Process all pairs in folder (fast mode)
        face-morph batch data/faces/

        # Full mode with GPU acceleration
        face-morph batch data/faces/ --full --gpu

        # Custom output directory
        face-morph batch data/ -o batch_results/
    """
    from datetime import datetime

    # Discover mesh files
    try:
        mesh_files = discover_meshes(folder)
        pairs = list(combinations(mesh_files, 2))
    except Exception as e:
        click.secho(f"\n✗ Error discovering meshes: {e}", fg='red', bold=True)
        sys.exit(1)

    if len(pairs) == 0:
        click.secho(f"\n✗ No valid mesh pairs found in {folder}", fg='red', bold=True)
        sys.exit(1)

    # Shared timestamp for all pairs in this batch
    batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    click.echo()
    click.secho(f"Found {len(mesh_files)} meshes, processing {len(pairs)} unique pairs...", fg='cyan', bold=True)
    click.echo()

    # Process each pair
    failed_pairs = []
    for idx, (mesh1, mesh2) in enumerate(pairs, 1):
        click.secho(f"[{idx}/{len(pairs)}] {mesh1.name} + {mesh2.name}", fg='cyan', bold=True)

        try:
            config = MorphConfig(
                input_mesh_1=mesh1,
                input_mesh_2=mesh2,
                output_dir=output or Path('results'),
                output_mode='full' if full else 'default',
                device=torch.device('cuda' if gpu else 'cpu'),
                verbose=verbose,
                log_level=log_level.upper(),
                timestamp=batch_timestamp  # Share timestamp across batch
            )

            run_morphing_pipeline(config)
            click.secho(f"  ✓ Completed\n", fg='green')

        except KeyboardInterrupt:
            click.echo()
            click.secho("\n✗ Batch interrupted by user", fg='yellow')
            sys.exit(130)

        except Exception as e:
            click.secho(f"  ✗ Failed: {e}\n", fg='red')
            failed_pairs.append((mesh1.name, mesh2.name))
            continue

    # Summary
    click.echo()
    click.secho("="*70, fg='cyan')
    click.secho("BATCH COMPLETE", fg='cyan', bold=True)
    click.secho("="*70, fg='cyan')
    click.echo(f"Total pairs: {len(pairs)}")
    click.echo(f"Successful: {len(pairs) - len(failed_pairs)}")
    click.echo(f"Failed: {len(failed_pairs)}")

    if failed_pairs:
        click.echo()
        click.secho("Failed pairs:", fg='red')
        for mesh1, mesh2 in failed_pairs:
            click.echo(f"  - {mesh1} + {mesh2}")

    click.echo()


@click.group()
@click.version_option(version='1.0.0', prog_name='face-morph')
def cli():
    """
    Face Morphing - Production-ready 3D face morphing with GPU acceleration.

    \b
    Output Modes:
      Default: PNG images + heatmaps (fast, ~5-10 min)
      Full:    PNG + heatmaps + meshes + video + CSV (complete, ~2 hours)

    \b
    Examples:
        # Morph two faces (default mode)
        face-morph morph face1.fbx face2.fbx

        # Full output with GPU
        face-morph morph face1.fbx face2.fbx --full --gpu

        # Batch process folder
        face-morph batch data/faces/ --full --gpu

    \b
    For more help on a specific command:
        face-morph morph --help
        face-morph batch --help
    """
    pass


def discover_meshes(folder: Path) -> list:
    """
    Discover all mesh files in a folder.

    Prefers FBX over OBJ when both exist for same mesh.

    Args:
        folder: Directory containing mesh files

    Returns:
        List of unique mesh file paths

    Raises:
        ValueError: If folder contains no valid mesh files
    """
    # Find all mesh files
    all_files = list(folder.iterdir())

    # Group by stem (filename without extension)
    mesh_dict = {}
    for f in all_files:
        if f.suffix.lower() in {'.fbx', '.obj'}:
            stem = f.stem
            # Prefer FBX over OBJ
            if stem not in mesh_dict or f.suffix.lower() == '.fbx':
                mesh_dict[stem] = f

    mesh_files = sorted(mesh_dict.values())

    if len(mesh_files) < 2:
        raise ValueError(
            f"Need at least 2 unique mesh files in {folder}\n"
            f"Found: {len(mesh_files)} files"
        )

    return mesh_files


# Register commands
cli.add_command(morph)
cli.add_command(batch)


def main():
    """Entry point for console_scripts."""
    cli()


if __name__ == '__main__':
    main()
