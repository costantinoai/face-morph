"""
Format Conversion
=================

Single responsibility: Convert between 3D file formats.
"""

import subprocess
import tempfile
from pathlib import Path


def convert_fbx_to_obj(
    fbx_path: Path,
    obj_path: Path,
    blender_path: str = 'blender',
    force: bool = False
) -> bool:
    """
    Convert FBX file to OBJ format using Blender.

    Single responsibility: Handle FBX→OBJ conversion.

    Args:
        fbx_path: Source FBX file
        obj_path: Target OBJ file
        blender_path: Path to Blender executable
        force: Force conversion even if OBJ exists

    Returns:
        True if successful or cached, False if failed
    """
    # Use cached OBJ if exists
    if obj_path.exists() and not force:
        return True

    # Enhanced Blender script with texture extraction
    conversion_script = f"""
import bpy
import sys
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX
try:
    bpy.ops.import_scene.fbx(filepath="{fbx_path}")
except:
    sys.exit(1)

# Find meshes
meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
if not meshes:
    sys.exit(1)

# Extract and save textures
output_dir = os.path.dirname("{obj_path}")
base_name = os.path.splitext(os.path.basename("{obj_path}"))[0]

# Process all loaded images (textures from FBX)
for img in bpy.data.images:
    if img.source == 'FILE' or img.packed_file:
        # Save image as PNG in output directory
        texture_path = os.path.join(output_dir, f"{{base_name}}_{{img.name}}.png")

        # If image is packed (embedded), unpack it first
        if img.packed_file:
            img.filepath_raw = texture_path
            img.file_format = 'PNG'
            try:
                img.save()
            except:
                pass
        # If image has a filepath, copy/save it
        elif img.filepath:
            img.filepath_raw = texture_path
            img.file_format = 'PNG'
            try:
                img.save()
            except:
                pass

# Update material texture references to point to saved PNGs
for mat in bpy.data.materials:
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                img = node.image
                # Update to point to our saved PNG
                texture_filename = f"{{base_name}}_{{img.name}}.png"
                node.image.filepath = texture_filename

# Export OBJ
bpy.ops.object.select_all(action='DESELECT')
for obj in meshes:
    obj.select_set(True)

try:
    # Try Blender 4.0+ API
    bpy.ops.wm.obj_export(
        filepath="{obj_path}",
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        path_mode='RELATIVE'
    )
except AttributeError:
    # Fallback to old API
    bpy.ops.export_scene.obj(
        filepath="{obj_path}",
        use_selection=True,
        use_materials=True,
        use_uvs=True,
        use_normals=True,
        path_mode='RELATIVE'
    )
"""

    # Write temporary script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(conversion_script)
        script_path = f.name

    try:
        # Run Blender
        result = subprocess.run(
            [blender_path, '--background', '--python', script_path],
            capture_output=True,
            timeout=120
        )

        # Cleanup
        Path(script_path).unlink(missing_ok=True)

        # Check success
        return result.returncode == 0 and obj_path.exists()

    except (FileNotFoundError, subprocess.TimeoutExpired):
        Path(script_path).unlink(missing_ok=True)
        return False


def convert_obj_to_fbx(
    obj_path: Path,
    fbx_path: Path,
    texture_path: Path = None,
    blender_path: str = 'blender'
) -> bool:
    """
    Convert OBJ file to FBX format using Blender.

    Single responsibility: Handle OBJ→FBX conversion.

    Args:
        obj_path: Source OBJ file
        fbx_path: Target FBX file
        texture_path: Optional texture to embed
        blender_path: Path to Blender executable

    Returns:
        True if successful, False if failed
    """
    # Blender script for OBJ to FBX conversion
    conversion_script = f"""
import bpy
import sys

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ
try:
    # Try Blender 4.0+ API
    bpy.ops.wm.obj_import(filepath="{obj_path}")
except AttributeError:
    # Fallback to old API (Blender < 4.0)
    try:
        bpy.ops.import_scene.obj(filepath="{obj_path}")
    except:
        sys.exit(1)
except:
    sys.exit(1)

# Find meshes
meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
if not meshes:
    sys.exit(1)

# Export FBX
try:
    # Try Blender 4.0+ API
    bpy.ops.export_scene.fbx(
        filepath="{fbx_path}",
        use_selection=False,
        global_scale=1.0,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z',
        axis_up='Y',
        embed_textures=True
    )
except Exception as e:
    print(f"Export failed: {{e}}")
    sys.exit(1)
"""

    # Write temporary script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(conversion_script)
        script_path = f.name

    try:
        # Run Blender
        result = subprocess.run(
            [blender_path, '--background', '--python', script_path],
            capture_output=True,
            timeout=120
        )

        # Cleanup
        Path(script_path).unlink(missing_ok=True)

        # Check success
        return result.returncode == 0 and fbx_path.exists()

    except (FileNotFoundError, subprocess.TimeoutExpired):
        Path(script_path).unlink(missing_ok=True)
        return False
