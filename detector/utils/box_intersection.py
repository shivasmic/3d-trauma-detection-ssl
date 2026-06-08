"""
3D Trauma Detection | Authors: Shivam Chaudhary, Sheethal Bhat, Andreas Maier | FAU Erlangen-Nürnberg
Copyright (c) 2026 | MIT License | https://github.com/shivasmic/3d-trauma-detection-ssl
"""

def box_intersection_fn(*args, **kwargs):
    raise NotImplementedError(
        "Cython box_intersection not compiled. "
        "This is only needed for rotated boxes. "
        "RSNA uses axis-aligned boxes and doesn't require this."
    )
