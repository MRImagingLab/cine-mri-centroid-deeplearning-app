# Cine MRI Centroid Predictor (DSNT Teaching App)

DSNT: Differentiable Spatial to Numerical Transform

This Streamlit app demonstrates how a DSNT-based neural network localizes
the **center of the heart region** in **full-FOV cine cardiac MRI**.

### Supported inputs
- Cine CMR images (PNG/JPG/TIF/BMP)
- MATLAB `.mat` files containing `image` (HxW or HxWxT)
- Views: 2CH / 3CH / 4CH / SAX (full FOV only)

### What the app shows
- Spatial probability map (softmax over logits)
- DSNT centroid (expectation)
- Optional argmax comparison
- Automatically sized ROI (image / 2)
- Intermediate feature maps (Only for teaching)
- Explanation of softmax β (not a threshold)


