import streamlit as st
import cv2
import numpy as np
import torch
import imageio
from PIL import Image

def estimate_depth(image_np):
    # Load MiDaS model
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    model.eval()
    
    # Preprocess image
    original_height, original_width = image_np.shape[:2]
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (384, 384))
    
    # Transform to tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Normalize for MiDaS
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Predict depth
    with torch.no_grad():
        depth = model(img_tensor)
    
    # Post-process depth map
    depth = depth.squeeze().cpu().numpy()
    depth = cv2.resize(depth, (original_width, original_height))
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    
    return depth_normalized

def create_animation(image_np, depth_map, num_frames=20, max_shift=50):
    frames = []
    height, width = image_np.shape[:2]
    
    # Create layers based on depth percentiles
    layers = []
    num_layers = 5
    depth_values = np.percentile(depth_map, np.linspace(0, 100, num_layers + 1))
    
    for i in range(num_layers):
        mask = np.logical_and(depth_map >= depth_values[i], 
                             depth_map <= depth_values[i+1])
        layers.append(mask)
    
    # Generate animation frames
    for frame in range(num_frames):
        progress = frame / num_frames
        shift = int(max_shift * np.sin(2 * np.pi * progress))
        
        composite = np.zeros_like(image_np)
        for layer_idx, mask in enumerate(layers):
            # Calculate layer shift (closer layers move more)
            layer_shift = shift * (layer_idx + 1) // 2
            
            # Shift layer
            shifted_layer = np.roll(image_np, layer_shift, axis=1)
            
            # Apply mask
            composite[mask] = shifted_layer[mask]
        
        frames.append(composite)
    
    return frames

def main():
    st.title("2D to 3D Animation Converter")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        
        # Estimate depth map
        depth_map = estimate_depth(image_np)
        
        # Normalize depth map for display
        depth_display = (depth_map * 255).astype(np.uint8)
        with col2:
            st.image(depth_display, caption="Depth Map")
        
        # Animation parameters
        st.sidebar.header("Animation Settings")
        duration = st.sidebar.slider("Duration (seconds)", 1, 10, 2)
        fps = st.sidebar.slider("FPS", 5, 30, 15)
        max_shift = st.sidebar.slider("Max Shift", 10, 100, 50)
        
        if st.button("Generate Animation"):
            num_frames = int(duration * fps)
            
            # Generate animation frames
            frames = create_animation(image_np, depth_map, 
                                      num_frames=num_frames, 
                                      max_shift=max_shift)
            
            # Convert frames to uint8
            frames = [frame.astype(np.uint8) for frame in frames]
            
            # Save as GIF
            with imageio.get_writer('animation.gif', mode='I', fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            # Display animation
            st.success("Animation generated!")
            st.image("animation.gif")

if __name__ == "__main__":
    main()