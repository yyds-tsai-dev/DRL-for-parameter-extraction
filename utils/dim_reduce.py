import numpy as np

def get_err_features(
    vgs: np.ndarray,
    concat_err_vector: np.ndarray,
    # i_meas_dict: dict,
    vto: float,
    vgo: float,
    n_curves: int,
) -> np.ndarray:
    """
    Extracts error features from the error vector based on the defined voltage regions.
    """

    len_single_curve = len(vgs)
    all_features = []

    sub_threshold_mask = vgs < vto
    knee_mask = (vgs >= vto) & (vgs < vgo)
    saturation_mask = vgs >= vgo

    for i in range(n_curves):
        start_idx = i * len_single_curve
        end_idx = (i + 1) * len_single_curve
        err_vector_single_curve = concat_err_vector[start_idx:end_idx]

        err_sub = err_vector_single_curve[sub_threshold_mask] if np.any(sub_threshold_mask) else np.array([0])
        err_knee = err_vector_single_curve[knee_mask] if np.any(knee_mask) else np.array([0])
        err_sat = err_vector_single_curve[saturation_mask] if np.any(saturation_mask) else np.array([0])
        
        features_single_curve = [
            np.mean(err_sub),
            np.sqrt(np.mean(np.square(err_sub))),
            np.mean(err_knee),
            np.sqrt(np.mean(np.square(err_knee))),
            np.mean(err_sat),
            np.sqrt(np.mean(np.square(err_sat))),
        ]
        all_features.extend(features_single_curve)

    final_features = np.array(all_features, dtype=np.float32)
    # for compressing dynamic range
    final_features = (
        np.sign(final_features) * np.log1p(np.abs(final_features))
    ).astype(np.float32)

    return final_features
