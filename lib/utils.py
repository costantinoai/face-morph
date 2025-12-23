"""Utility functions for the face morphing library."""

# Global verbose flag
_VERBOSE = False


def set_verbose(enabled: bool):
    """Set global verbose mode."""
    global _VERBOSE
    _VERBOSE = enabled


def vprint(*args, **kwargs):
    """Print only if verbose mode is enabled."""
    if _VERBOSE:
        print(*args, **kwargs)
