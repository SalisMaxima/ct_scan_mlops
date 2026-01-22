We assessed robustness of the drift detection by simulating realistic data drift through Gaussian noise augmentation of the training images. This reflects variations that may arise from different CT scanners, acquisition settings, or calibration changes across clinical sites.

Three scenarios were evaluated: low noise, high noise, and a positive mean shift.
Low-variance noise  did not trigger drift for most features, indicating robustness to minor sensor fluctuations. In contrast, high-variance noise and mean shifts resulted in clear drift detection in multiple intensity-based features (mean, percentiles, max), while spatial features (height, width) remained stable.

These results show that the monitoring setup is robust to small perturbations, yet sensitive to clinically relevant distribution shifts caused by changes in imaging conditions, validating its suitability for deployment.
