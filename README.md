Lane Detection Using U-Net

This project implements lane detection on road images using a U-Net convolutional neural network. It detects lane lines, highlights them on the original image, and provides a binary lane mask. The app is deployed using Streamlit for easy interaction.

Note: This model works best on clean, clear images. It may not perform well on blurry, low-light, or heavily occluded images.

Features

Binary Lane Segmentation: Predicts lane masks for input images.

Lane Highlighting: Draws left and right lane lines and fills the lane area with color.

Interactive Deployment: Run the project via a Streamlit web app.

Custom Image Input: Upload your own images and see lane detection results.

Project Structure
LaneDetection/
│
├─ lane_unet_model.h5        # Trained U-Net model
├─ lane_dec_deploy.py        # Streamlit application
├─ build model.py            # Lane detection scripts (model loading, mask prediction, highlighting)
├─ requirement.txt           # Python dependencies
├─ README.md                 # Project description
└─ lane_data_img/            # Sample dataset
   ├─ frames/                # Original road images
   └─ lane-masks/            # Corresponding lane masks
