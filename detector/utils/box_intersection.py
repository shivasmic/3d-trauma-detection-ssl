def box_intersection_fn(*args, **kwargs):
    raise NotImplementedError(
        "Cython box_intersection not compiled. "
        "This is only needed for rotated boxes. "
        "RSNA uses axis-aligned boxes and doesn't require this."
    )
