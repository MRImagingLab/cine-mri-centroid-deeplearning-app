Cine MRI Centroid Predictor (Teaching App)

This Streamlit application demonstrates how a neural network localizes the center of the cardiac region in full–field-of-view (FOV) cine cardiac MRI using a probability-based spatial representation.

The app is designed for education and visualization, illustrating how modern deep learning models convert spatial evidence into a stable numerical centroid without heuristic thresholding or post-hoc rules.

Supported Inputs

Cine CMR images: PNG / JPG / TIF / BMP

MATLAB .mat files containing:

image (HxW or HxWxT)

Cardiac views:

2CH, 3CH, 4CH, SAX (full FOV only)

What the App Demonstrates

Spatial probability map generated from network outputs

Centroid estimation computed as the probability-weighted spatial expectation

Optional argmax comparison to illustrate instability of peak-based localization

Automatically scaled ROI (default: half the image size)

Intermediate feature maps (enabled for teaching and inspection)

Softmax sharpness parameter (β) explained as a distribution control, not a threshold

Educational Focus

Highlights why probability-weighted localization is more stable than hard peak detection

Demonstrates robustness to noise, contrast variation, and temporal changes in cine data

Emphasizes fully differentiable, end-to-end learning suitable for medical imaging workflows

Intended Use

Teaching cardiac imaging trainees and engineers

Demonstrating modern centroid localization concepts in medical AI

Exploring differences between probabilistic and peak-based localization

🔗 Live App
https://cine-mri-centroid-deeplearning-app-8dsf6ayqtam7jqsxiskpxm.streamlit.app/
