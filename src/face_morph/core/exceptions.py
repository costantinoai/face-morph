"""Custom exceptions for face morphing operations.

This module defines domain-specific exceptions that provide clear,
actionable error messages for common failure modes in face morphing.
"""


class FaceMorphError(Exception):
    """Base exception for all face morphing errors.

    All custom exceptions in the face_morph package inherit from this base class.
    This allows catching all face_morph-related errors with a single except clause.

    Example:
        >>> try:
        ...     morph_mesh(mesh1, mesh2, ratio=0.5)
        ... except FaceMorphError as e:
        ...     print(f"Morphing failed: {e}")
    """
    pass


class TopologyMismatchError(FaceMorphError):
    """Raised when meshes have incompatible topology for morphing.

    Morphing requires meshes to have identical topology (same number of vertices
    and faces with consistent connectivity). This error is raised when attempting
    to morph meshes with different vertex or face counts.

    Attributes:
        verts1: Number of vertices in first mesh
        verts2: Number of vertices in second mesh
        faces1: Number of faces in first mesh (optional)
        faces2: Number of faces in second mesh (optional)

    Example:
        >>> if mesh1.num_verts != mesh2.num_verts:
        ...     raise TopologyMismatchError(mesh1.num_verts, mesh2.num_verts)
        TopologyMismatchError: Topology mismatch: mesh1 has 10,523 vertices,
        mesh2 has 8,941 vertices. Morphing requires identical topology.
    """

    def __init__(self, verts1: int, verts2: int, faces1: int = None, faces2: int = None):
        """Initialize topology mismatch error.

        Args:
            verts1: Vertex count in first mesh
            verts2: Vertex count in second mesh
            faces1: Face count in first mesh (optional)
            faces2: Face count in second mesh (optional)
        """
        self.verts1 = verts1
        self.verts2 = verts2
        self.faces1 = faces1
        self.faces2 = faces2

        # Build detailed error message
        msg = (
            f"Topology mismatch: mesh1 has {verts1:,} vertices, "
            f"mesh2 has {verts2:,} vertices"
        )

        if faces1 is not None and faces2 is not None:
            msg += f" (faces: {faces1:,} vs {faces2:,})"

        msg += ". Morphing requires identical topology."

        super().__init__(msg)


class TextureMissingError(FaceMorphError):
    """Raised when a required texture is missing or invalid.

    This error is raised when texture data is expected but not found,
    or when texture dimensions don't match requirements.

    Example:
        >>> if texture is None and require_texture:
        ...     raise TextureMissingError("Mesh has no texture but texture morphing was requested")
    """
    pass


class RendererError(FaceMorphError):
    """Raised when rendering operations fail.

    This can occur due to:
    - Invalid mesh data
    - GPU memory issues
    - OpenGL/PyRender initialization failures
    - Invalid render parameters

    Example:
        >>> try:
        ...     renderer.render_mesh(mesh)
        ... except Exception as e:
        ...     raise RendererError(f"Failed to render mesh: {e}") from e
    """
    pass


class CUDANotAvailableError(FaceMorphError):
    """Raised when CUDA is requested but not available.

    This error provides helpful diagnostics and suggestions when GPU
    acceleration is requested but CUDA is not properly configured.

    Example:
        >>> if device.type == 'cuda' and not torch.cuda.is_available():
        ...     raise CUDANotAvailableError()
        CUDANotAvailableError: CUDA requested but not available.
        ...
    """

    def __init__(self, details: str = None):
        """Initialize CUDA not available error.

        Args:
            details: Optional additional context about the CUDA issue
        """
        msg = (
            "CUDA requested but not available.\n"
            "Options:\n"
            "  1. Use CPU mode: set device to 'cpu'\n"
            "  2. Install CUDA: https://developer.nvidia.com/cuda-downloads\n"
            "  3. Reinstall PyTorch with CUDA support:\n"
            "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
        )

        if details:
            msg = f"{msg}\n\nDetails: {details}"

        super().__init__(msg)


class MeshLoadError(FaceMorphError):
    """Raised when mesh loading fails.

    Common causes:
    - File not found
    - Unsupported file format
    - Corrupted mesh data
    - Missing texture references

    Example:
        >>> try:
        ...     mesh = load_mesh(path)
        ... except Exception as e:
        ...     raise MeshLoadError(f"Failed to load {path}: {e}") from e
    """
    pass


class MeshSaveError(FaceMorphError):
    """Raised when mesh saving fails.

    Common causes:
    - Invalid output path
    - Permission denied
    - Disk full
    - Invalid mesh data

    Example:
        >>> try:
        ...     save_mesh(mesh, output_path)
        ... except Exception as e:
        ...     raise MeshSaveError(f"Failed to save to {output_path}: {e}") from e
    """
    pass


class ValidationError(FaceMorphError):
    """Raised when input validation fails.

    Used for general input validation failures where a more specific
    exception type doesn't exist.

    Example:
        >>> if ratio < 0 or ratio > 1:
        ...     raise ValidationError(f"Ratio must be in [0, 1], got {ratio}")
    """
    pass


class ConversionError(FaceMorphError):
    """Raised when format conversion fails.

    Used for OBJ ↔ FBX conversion failures via Blender.

    Example:
        >>> if not convert_obj_to_fbx(obj_path, fbx_path):
        ...     raise ConversionError(f"Failed to convert {obj_path} to FBX")
    """
    pass
