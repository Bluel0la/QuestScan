PREPROCESSING_PROFILES = [
    # Default
    {
        "denoise_ksize": 3,
        "threshold_C": 2,
        "clahe_clip_limit": 2.0,
    },
    # More aggressive contrast
    {
        "denoise_ksize": 3,
        "threshold_C": 1,
        "clahe_clip_limit": 3.0,
    },
    # Stronger denoise, looser threshold
    {
        "denoise_ksize": 5,
        "threshold_C": 3,
        "clahe_clip_limit": 2.5,
    },
]
