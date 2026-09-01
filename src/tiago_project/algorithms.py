import numpy as np

def ransac_linear_regression(x, y, max_iters=100, threshold=0.2):
    """
    Pure NumPy RANSAC to fit y = s*x + t, ignoring outliers.
    Compatible with Python 2.7 and older NumPy versions.
    """
    best_s = 1.0
    best_t = 0.0
    max_inliers = 0
    n_points = len(x)

    if n_points < 2:
        return best_s, best_t

    for _ in range(max_iters): # use xrange(max_iters) if very strict, but range() is fine here
        # Randomly select 2 points
        idx = np.random.choice(n_points, 2, replace=False)
        x_sample = x[idx]
        y_sample = y[idx]

        # Prevent division by zero if x values are identical
        if abs(x_sample[1] - x_sample[0]) < 1e-6:
            continue

        # Calculate line parameters (s, t) for these 2 points
        s = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0])
        t = y_sample[0] - s * x_sample[0]

        # Count how many points agree with this line (inliers)
        y_est = s * x + t
        errors = np.abs(y - y_est)
        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)

        # Save the best model
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            best_s = s
            best_t = t

    # Final polish: Re-run a standard polyfit using ONLY the inliers
    final_errors = np.abs(y - (best_s * x + best_t))
    final_inlier_mask = final_errors < threshold
    if np.sum(final_inlier_mask) > 10:
        best_s, best_t = np.polyfit(x[final_inlier_mask], y[final_inlier_mask], 1)

    return best_s, best_t
