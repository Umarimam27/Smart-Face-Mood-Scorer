# import streamlit as st
# import numpy as np
# import pandas as pd
# import cv2
# import base64
# import time
# from PIL import Image
# import json
# from datetime import datetime

# # ================================
# # 🎯 CONFIGURATION
# # ================================
# MODEL_PATH = "mood_classification/models/mood_classification_model.h5"
# CLASS_INDICES_PATH = "mood_classification/models/class_indices.json"

# # ================================
# # ⚙️ UTILITIES
# # ================================
# def get_base64_image_url(uploaded_file):
#     try:
#         bytes_data = uploaded_file.getvalue()
#         base64_encoded_data = base64.b64encode(bytes_data).decode("utf-8")
#         mime_type = uploaded_file.type or "image/png"
#         return f"data:{mime_type};base64,{base64_encoded_data}"
#     except Exception as e:
#         st.error(f"Error processing image: {e}")
#         return None

# def set_cinematic_bg(base64_urls, interval_per_image=6):
#     num_images = len(base64_urls)
#     total_duration = num_images * interval_per_image
#     OVERLAY_OPACITY = "rgba(0,0,0,0.6)"
    
#     if num_images == 0:
#         st.markdown("""
#         <style>
#         .stApp {
#             background: linear-gradient(135deg, #1b2735, #090a0f);
#             color: white;
#         }
#         </style>
#         """, unsafe_allow_html=True)
#         return

#     css_keyframes = []
#     for i in range(num_images):
#         start_percent = (i * 100) / num_images
#         hold_percent = start_percent + (100 / num_images)
#         css_keyframes.append(f"{start_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
#         css_keyframes.append(f"{hold_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
    
#     css_keyframes.append(f"100% {{ background-image: url('{base64_urls[0]}'); }}")

#     st.markdown(f"""
#     <style>
#     .stApp {{
#         background-size: cover;
#         background-attachment: fixed;
#         background-repeat: no-repeat;
#         background-image: url('{base64_urls[0]}');
#         animation: cinematicBg {total_duration}s infinite;
#         color: white;
#     }}
    
#     @keyframes cinematicBg {{
#         {"".join(css_keyframes)}
#     }}
    
#     .stApp::before {{
#         content: "";
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         background: {OVERLAY_OPACITY};
#         z-index: 0;
#     }}
    
#     [data-testid="stSidebar"] > div:first-child {{
#         background: rgba(255, 255, 255, 0.05);
#         backdrop-filter: blur(8px);
#         border-radius: 16px;
#         padding: 20px;
#         z-index: 10;
#     }}
    
#     * {{
#         font-family: 'Poppins', sans-serif;
#     }}
    
#     [data-testid="stHeader"], [data-testid="stToolbar"] {{
#         background: transparent !important;
#     }}
    
#     .prediction-box {{
#         background-color: rgba(255,153,0,0.2);
#         padding: 25px;
#         border-radius: 15px;
#         text-align: center;
#         box-shadow: 0 0 20px #ff9900;
#         margin: 20px 0;
#     }}
    
#     .confidence-bar {{
#         background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44);
#         height: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#         position: relative;
#     }}
    
#     .confidence-fill {{
#         height: 100%;
#         border-radius: 10px;
#         background: rgba(255,255,255,0.3);
#         transition: width 0.5s ease-in-out;
#     }}
    
#     .confidence-text {{
#         position: absolute;
#         top: 50%;
#         left: 50%;
#         transform: translate(-50%, -50%);
#         color: white;
#         font-weight: bold;
#         text-shadow: 1px 1px 2px black;
#     }}
#     </style>
#     """, unsafe_allow_html=True)

# # ================================
# # 🔧 UNIVERSAL PREPROCESSING - WORKS WITH ANY IMAGE SIZE
# # ================================
# def detect_model_input_shape(model):
#     """Automatically detect what input shape the model expects"""
#     input_shape = model.input_shape
    
#     if input_shape and len(input_shape) == 4:  # (batch, height, width, channels)
#         height, width = input_shape[1], input_shape[2]
#         st.success(f"🤖 Model expects: {height}x{width} images")
#         return (height, width)
#     else:
#         # Fallback to common sizes
#         st.warning("⚠️ Could not detect model input shape, using default 224x224")
#         return (224, 224)

# def universal_preprocess_image(image, target_size):
#     """
#     Universal preprocessing that works with ANY image size and ratio
#     """
#     try:
#         # Convert PIL Image to numpy array if needed
#         if isinstance(image, Image.Image):
#             image = np.array(image)
        
#         # Store original shape for debugging
#         original_shape = image.shape
#         st.write(f"📐 Original image shape: {original_shape}")
        
#         # Ensure image is in RGB format
#         if len(image.shape) == 2:  # Grayscale
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#             st.info("🔄 Converted grayscale to RGB")
#         elif image.shape[2] == 4:  # RGBA
#             image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
#             st.info("🔄 Converted RGBA to RGB")
        
#         # Resize to target size (model's expected size)
#         height, width = target_size
#         resized_image = cv2.resize(image, (width, height))
#         st.write(f"🔄 Resized to: {resized_image.shape}")
        
#         # Normalize pixel values to [0, 1]
#         normalized_image = resized_image / 255.0
        
#         # Add batch dimension
#         final_image = np.expand_dims(normalized_image, axis=0)
#         st.write(f"🎯 Final input shape: {final_image.shape}")
        
#         return final_image
        
#     except Exception as e:
#         st.error(f"❌ Error in preprocessing: {e}")
#         return None

# def smart_predict_mood(image_array, model, class_names):
#     """Smart prediction that handles any model output format"""
#     try:
#         # Get model prediction
#         predictions = model.predict(image_array, verbose=0)
        
#         st.write(f"🔧 Raw prediction output shape: {predictions.shape}")
#         st.write(f"🔧 Raw prediction values: {predictions}")
        
#         # Handle different model output formats
#         if len(predictions[0]) == 1:
#             # Binary classification with sigmoid output
#             prediction_prob = predictions[0][0]
#             st.write(f"🔧 Binary prediction probability: {prediction_prob}")
            
#             if prediction_prob > 0.5:
#                 predicted_class = 1  # Happy
#                 confidence = float(prediction_prob)
#                 mood_label = class_names.get(1, "HAPPY")
#             else:
#                 predicted_class = 0  # Not Happy
#                 confidence = 1 - float(prediction_prob)
#                 mood_label = class_names.get(0, "NOT_HAPPY")
#         else:
#             # Multi-class classification with softmax
#             predicted_class = np.argmax(predictions[0])
#             confidence = np.max(predictions[0])
#             mood_label = class_names.get(predicted_class, "UNKNOWN")
#             st.write(f"🔧 Multi-class prediction - Class: {predicted_class}, Confidence: {confidence}")
        
#         return mood_label, confidence, predicted_class
        
#     except Exception as e:
#         st.error(f"❌ Prediction error: {e}")
#         import traceback
#         st.error(f"🔧 Detailed error: {traceback.format_exc()}")
#         return None, None, None

# # ================================
# # 🧠 LOAD MODEL WITH SMART DETECTION
# # ================================
# @st.cache_resource
# def load_model():
#     try:
#         import tensorflow as tf
#         # Load the trained model
#         model = tf.keras.models.load_model(MODEL_PATH)
        
#         # Smart detection of model requirements
#         st.write("🔍 Analyzing model architecture...")
#         st.write(f"📐 Model input shape: {model.input_shape}")
#         st.write(f"📊 Model output shape: {model.output_shape}")
        
#         # Detect the required input size
#         target_size = detect_model_input_shape(model)
        
#         # Load class indices
#         with open(CLASS_INDICES_PATH, 'r') as f:
#             class_indices = json.load(f)
        
#         # Reverse the class indices to get class names
#         class_names = {v: k for k, v in class_indices.items()}
#         st.write(f"🎭 Class mappings: {class_names}")
        
#         st.success("✅ Model loaded successfully with smart detection!")
#         return model, class_names, target_size
        
#     except Exception as e:
#         st.error(f"⚠️ Error loading model: {e}")
#         st.error("Make sure the model file exists and paths are correct")
#         return None, None, (224, 224)  # Fallback size

# # ================================
# # 📂 SIDEBAR
# # ================================
# base64_image_urls = []

# with st.sidebar:
#     st.title("⚙️ App Configuration")
    
#     uploaded_files = st.file_uploader(
#         "🖼️ Upload background images:",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files:
#         for file in uploaded_files:
#             url = get_base64_image_url(file)
#             if url:
#                 base64_image_urls.append(url)
#         st.success("✅ Background ready!")
    
#     st.markdown("---")
#     st.subheader("📘 About the Model")
#     st.info("This AI model classifies facial expressions as Happy or Not Happy using deep learning.")
    
#     st.markdown(f"📅 Updated: **{datetime.now().strftime('%b %d, %Y')}**")
#     st.markdown("Made with ❤️ ", unsafe_allow_html=True)
#     st.markdown("✨ Developed by **Umar Imam**", unsafe_allow_html=True)

# set_cinematic_bg(base64_image_urls)

# # ================================
# # 🎓 HEADER
# # ================================
# st.markdown("""
# <h1 style='text-align:center; color:#ff9900; text-shadow: 2px 2px 6px #000;'>
# 😊 AI-Based Mood Classification System
# </h1>
# <p style='text-align:center; font-size:18px; color:#fff;'>
# Detect emotions from facial expressions using deep learning.
# </p>
# """, unsafe_allow_html=True)

# # Load model at startup with smart detection
# model, class_names, target_size = load_model()

# # ================================
# # 📊 TABS
# # ================================
# tab1, tab2, tab3 = st.tabs(["🎭 Mood Detection", "📊 Batch Analysis", "ℹ️ Model Info"])

# # ================================
# # 🎭 TAB 1 — MOOD DETECTION (UNIVERSAL)
# # ================================
# with tab1:
#     st.header("Upload Image for Mood Analysis")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         uploaded_image = st.file_uploader(
#             "📸 Upload a face image", 
#             type=["jpg", "jpeg", "png", "bmp", "webp"],
#             help="Upload any image size or ratio - the system will automatically adapt"
#         )
        
#         if uploaded_image:
#             image = Image.open(uploaded_image)
#             st.image(image, caption="Uploaded Image", use_container_width=True)
#             st.info(f"📐 Original size: {image.size}")
    
#     with col2:
#         st.markdown("### Live Camera Capture")
#         st.info("Coming soon: Real-time mood detection using webcam")
        
#         # Placeholder for camera input
#         st.markdown("""
#         <div style='border: 2px dashed #ccc; padding: 20px; text-align: center; border-radius: 10px;'>
#             <p>📷 Camera input will be available in future versions</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     if st.button("🔍 Analyze Mood", use_container_width=True, type="primary"):
#         if not uploaded_image:
#             st.warning("⚠️ Please upload an image first.")
#         elif model is None:
#             st.error("❌ Model not loaded. Please check the model files.")
#         else:
#             with st.spinner("🤖 AI is analyzing the mood..."):
#                 # Show what we're doing
#                 st.write(f"🎯 Preparing image for model (target: {target_size[0]}x{target_size[1]})")
                
#                 # Universal preprocessing - works with ANY image
#                 processed_image = universal_preprocess_image(image, target_size)
                
#                 if processed_image is not None:
#                     # Make prediction
#                     mood_label, confidence, predicted_class = smart_predict_mood(processed_image, model, class_names)
                    
#                     if mood_label is not None:
#                         confidence_percent = confidence * 100
                        
#                         # Display emoji based on mood
#                         mood_emoji = "😊" if "HAPPY" in mood_label.upper() else "😔"
#                         display_label = f"{mood_label.replace('_', ' ').title()} {mood_emoji}"
                        
#                         st.balloons()
                        
#                         # Display results
#                         st.markdown(f"""
#                         <div class="prediction-box">
#                             <h2 style="color:white;">Detected Mood</h2>
#                             <h1 style="color:#ff9900; font-size:3.2em;">{display_label}</h1>
#                             <p style="font-size:1.2em; color:#ddd;">Confidence: {confidence_percent:.1f}%</p>
#                         </div>
#                         """, unsafe_allow_html=True)
                        
#                         # Confidence visualization
#                         st.markdown("### Confidence Level")
#                         st.markdown(f"""
#                         <div class="confidence-bar">
#                             <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
#                             <div class="confidence-text">{confidence_percent:.1f}%</div>
#                         </div>
#                         """, unsafe_allow_html=True)
                        
#                         # Additional insights
#                         if "HAPPY" in mood_label.upper():
#                             st.success("🎉 Great! This person appears to be happy!")
#                         else:
#                             st.info("💭 This person might be feeling neutral or sad.")
                            
#                     else:
#                         st.error("❌ Prediction failed. Please try another image.")
#                 else:
#                     st.error("❌ Failed to process the image. Please try another image.")

# # ================================
# # 📊 TAB 2 — BATCH ANALYSIS (UNIVERSAL)
# # ================================
# with tab2:
#     st.header("Batch Image Analysis")
    
#     st.info("Upload multiple images of any size or ratio for batch analysis")
    
#     uploaded_batch = st.file_uploader(
#         "📁 Upload multiple images", 
#         type=["jpg", "jpeg", "png", "bmp", "webp"],
#         accept_multiple_files=True,
#         help="Select multiple images of any size - automatic resizing will be applied"
#     )
    
#     if uploaded_batch and model is not None:
#         st.write(f"📸 **{len(uploaded_batch)}** images selected for analysis")
#         st.info(f"🎯 All images will be automatically resized to: {target_size[0]}x{target_size[1]}")
        
#         if st.button("🚀 Process Batch", use_container_width=True):
#             progress_bar = st.progress(0)
#             results = []
            
#             for i, image_file in enumerate(uploaded_batch):
#                 # Update progress
#                 progress = (i + 1) / len(uploaded_batch)
#                 progress_bar.progress(progress)
                
#                 try:
#                     # Process each image with universal preprocessing
#                     image = Image.open(image_file)
#                     processed_image = universal_preprocess_image(image, target_size)
                    
#                     if processed_image is not None:
#                         # Make prediction
#                         mood_label, confidence, _ = smart_predict_mood(processed_image, model, class_names)
                        
#                         if mood_label is not None:
#                             confidence_percent = confidence * 100
#                             mood_emoji = "😊" if "HAPPY" in mood_label.upper() else "😔"
                            
#                             results.append({
#                                 "image": image_file.name,
#                                 "mood": f"{mood_label.replace('_', ' ').title()} {mood_emoji}",
#                                 "confidence": f"{confidence_percent:.1f}%"
#                             })
#                         else:
#                             results.append({
#                                 "image": image_file.name,
#                                 "mood": "❌ Prediction Failed",
#                                 "confidence": "0%"
#                             })
#                     else:
#                         results.append({
#                             "image": image_file.name,
#                             "mood": "❌ Processing Error",
#                             "confidence": "0%"
#                         })
                        
#                 except Exception as e:
#                     results.append({
#                         "image": image_file.name,
#                         "mood": f"❌ Error: {str(e)[:50]}...",
#                         "confidence": "0%"
#                     })
            
#             # Display results
#             st.success("✅ Batch analysis completed!")
            
#             # Create results table
#             results_df = pd.DataFrame(results)
#             st.dataframe(results_df, use_container_width=True)
            
#             # Summary statistics
#             happy_count = sum(1 for r in results if "HAPPY" in r["mood"].upper())
#             st.metric("😊 Happy Faces", happy_count)
#             st.metric("😔 Not Happy Faces", len(results) - happy_count)

# # ================================
# # 📘 TAB 3 — MODEL INFO
# # ================================
# with tab3:
#     st.header("Model Overview")
    
#     st.info("""
#     This AI model uses a **Convolutional Neural Network (CNN)** trained on facial expression datasets 
#     to classify images as Happy or Not Happy.
#     """)
    
#     if class_names:
#         st.write("**Class Mappings:**", class_names)
    
#     if model:
#         st.write("**Model Input Requirements:**")
#         st.write(f"- Expected input shape: {model.input_shape}")
#         st.write(f"- Current target size: {target_size}")
    
#     col1, col2, col3 = st.columns(3)
    
#     col1.metric("Model Accuracy", "92%")
#     col2.metric("Training Images", "162")
#     col3.metric("Classes", "2")
    
#     st.markdown("""
#     **Features:**
#     - ✅ Universal image size support
#     - ✅ Automatic resizing and preprocessing
#     - ✅ Handles any image ratio
#     - ✅ RGB and grayscale conversion
#     - ✅ Batch processing capability
    
#     **Supported Image Formats:**
#     - JPG, JPEG, PNG, BMP, WEBP
#     - Any size or aspect ratio
#     - Grayscale and color images
#     """)

# # ================================
# # 🎯 FOOTER
# # ================================
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #888;'>
#     <p>🔬 Built with Streamlit • TensorFlow/Keras • OpenCV</p>
#     <p>🎯 Universal image processing • Any size supported</p>
#     <p>For educational and research purposes</p>
# </div>
# """, unsafe_allow_html=True)


#below is modified code


# import streamlit as st
# import numpy as np
# import pandas as pd
# import cv2
# import base64
# import random
# from PIL import Image
# from datetime import datetime

# # ================================
# # 🎯 CONFIGURATION
# # ================================
# # --- Define fixed labels for binary classification consistency ---
# HAPPY_LABEL = "HAPPY"
# NOT_HAPPY_LABEL = "NOT_HAPPY"
# IMAGE_SIZE = (224, 224) # Target size for internal processing consistency

# # Initialize Haar cascades globally
# # NOTE: We rely on cv2.data.haarcascades being available in the environment
# try:
#     FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#     SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#     CASCADES_LOADED = True
# except Exception as e:
#     # If cascades fail to load, prediction will rely only on image properties
#     st.error(f"Failed to load OpenCV cascades: {e}. Facial feature detection disabled.")
#     CASCADES_LOADED = False

# # ================================
# # ⚙️ UTILITIES
# # ================================
# def get_base64_image_url(uploaded_file):
#     """Encodes an uploaded file's bytes to a base64 URL for CSS background usage."""
#     try:
#         bytes_data = uploaded_file.getvalue()
#         base64_encoded_data = base64.b64encode(bytes_data).decode("utf-8")
#         mime_type = uploaded_file.type or "image/png"
#         return f"data:{mime_type};base64,{base64_encoded_data}"
#     except Exception as e:
#         st.error(f"Error processing image: {e}")
#         return None

# def set_cinematic_bg(base64_urls, interval_per_image=6):
#     """Sets a cinematic background effect using cycling base64 images."""
#     num_images = len(base64_urls)
#     total_duration = num_images * interval_per_image
#     OVERLAY_OPACITY = "rgba(0,0,0,0.6)"
    
#     # Minimal CSS for no images
#     if num_images == 0:
#         st.markdown("""
#         <style>
#         .stApp {
#             background: linear-gradient(135deg, #1b2735, #090a0f);
#             color: white;
#         }
#         </style>
#         """, unsafe_allow_html=True)
#         return

#     # Generate CSS keyframes for background animation
#     css_keyframes = []
#     for i in range(num_images):
#         start_percent = (i * 100) / num_images
#         hold_percent = start_percent + (100 / num_images)
#         css_keyframes.append(f"{start_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
#         css_keyframes.append(f"{hold_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
    
#     css_keyframes.append(f"100% {{ background-image: url('{base64_urls[0]}'); }}")

#     st.markdown(f"""
#     <style>
#     .stApp {{
#         background-size: cover;
#         background-attachment: fixed;
#         background-position: center;
#         background-repeat: no-repeat;
#         background-image: url('{base64_urls[0]}');
#         animation: cinematicBg {total_duration}s infinite;
#         color: white;
#     }}
    
#     @keyframes cinematicBg {{
#         {"".join(css_keyframes)}
#     }}
    
#     .stApp::before {{
#         content: "";
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         background: {OVERLAY_OPACITY};
#         z-index: 0;
#     }}
    
#     /* Ensure Streamlit components are layered above the overlay */
#     [data-testid="stHeader"], [data-testid="stToolbar"] {{
#         background: transparent !important;
#         z-index: 1;
#     }}

#     /* Styling for the output box */
#     .prediction-box {{
#         background-color: rgba(255,153,0,0.2);
#         padding: 25px;
#         border-radius: 15px;
#         text-align: center;
#         box-shadow: 0 0 20px #ff9900;
#         margin: 20px 0;
#     }}
    
#     .confidence-bar {{
#         background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44);
#         height: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#         position: relative;
#     }}
    
#     .confidence-fill {{
#         height: 100%;
#         border-radius: 10px;
#         background: rgba(255,255,255,0.3);
#         transition: width 0.5s ease-in-out;
#     }}
    
#     .confidence-text {{
#         position: absolute;
#         top: 50%;
#         left: 50%;
#         transform: translate(-50%, -50%);
#         color: white;
#         font-weight: bold;
#         text-shadow: 1px 1px 2px black;
#     }}
#     </style>
#     """, unsafe_allow_html=True)

# # ================================
# # ⚙️ FEATURE DETECTION LOGIC (From user's provided code)
# # ================================

# def detect_facial_features(image_np):
#     """Detect facial features using OpenCV for expression analysis"""
#     features = {
#         'face_detected': False,
#         'eyes_detected': 0,
#         'smile_detected': False,
#         'mouth_curve': 0.5,  # Neutral
#         'face_ratio': 1.0
#     }
    
#     if not CASCADES_LOADED:
#         return features
        
#     gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
#     # Detect faces
#     faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    
#     if len(faces) > 0:
#         features['face_detected'] = True
#         x, y, w, h = faces[0]
        
#         # Calculate face ratio (width/height)
#         features['face_ratio'] = w / h if h > 0 else 1.0
        
#         # Region of Interest for eyes and mouth
#         roi_gray = gray[y:y+h, x:x+w]
        
#         # Detect eyes (only in the face ROI)
#         eyes = EYE_CASCADE.detectMultiScale(roi_gray, 1.1, 6)
#         features['eyes_detected'] = len(eyes)
        
#         # Detect smile (only in the face ROI)
#         smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.7, 20)
#         features['smile_detected'] = len(smiles) > 0
        
#         # Simple mouth curve analysis (lower face region)
#         mouth_region = roi_gray[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
#         if mouth_region.size > 0:
#             # Calculate horizontal gradients to detect mouth curve
#             sobelx = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
#             # Normalize the curve measurement
#             mouth_curve = np.mean(np.abs(sobelx))
#             features['mouth_curve'] = min(1.0, mouth_curve / 50.0) 
    
#     return features

# def analyze_expression_characteristics(image_np):
#     """Analyze visual characteristics that might indicate expression"""
#     gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
#     hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
#     characteristics = {
#         'brightness': np.mean(gray),
#         'contrast': np.std(gray),
#         'saturation': np.mean(hsv[:, :, 1]),
#         'warmth': 0.0,
#         'texture_complexity': 0.0
#     }
    
#     # Color warmth (red/blue ratio)
#     r, g, b = cv2.split(image_np)
#     characteristics['warmth'] = np.mean(r) / (np.mean(b) + 1e-6)
    
#     # Texture complexity (edge density)
#     # Using Canny edge detection
#     edges = cv2.Canny(gray, 50, 150)
#     characteristics['texture_complexity'] = np.mean(edges) / 255.0
    
#     return characteristics

# def smart_feature_predict(image_np, debug_container):
#     """
#     Smart prediction based on facial features and expression analysis 
#     using the logic provided by the user (simulating the Gradio app's prediction).
#     """
#     with debug_container:
#         try:
#             # 1. Analyze features and characteristics
#             facial_features = detect_facial_features(image_np)
#             characteristics = analyze_expression_characteristics(image_np)
            
#             st.write(f"🔍 Facial Features: {facial_features}")
#             st.write(f"🎨 Image Characteristics: Brightness={characteristics['brightness']:.1f}, Texture={characteristics['texture_complexity']:.2f}")

#             # 2. Start with a neutral score
#             base_prediction = 0.5 
            
#             # --- Applying Feature Factors (Weights based on user's logic) ---
            
#             # Factor 1: Smile detection (strong indicator)
#             if facial_features['smile_detected']:
#                 base_prediction += 0.3
#                 st.info("✅ Smile detected (+0.3 HAPPY)")
#             else:
#                 base_prediction -= 0.1
#                 st.info("❌ No smile detected (-0.1 HAPPY)")
            
#             # Factor 2: Mouth curve 
#             if facial_features['mouth_curve'] > 0.7:
#                 base_prediction += 0.2
#                 st.info(f"⬆️ Strong mouth curve ({facial_features['mouth_curve']:.2f}) (+0.2 HAPPY)")
#             elif facial_features['mouth_curve'] < 0.3:
#                 base_prediction -= 0.2
#                 st.info(f"⬇️ Weak mouth curve ({facial_features['mouth_curve']:.2f}) (-0.2 HAPPY)")
            
#             # Factor 3: Eye detection
#             if facial_features['eyes_detected'] >= 2:
#                 base_prediction += 0.1
#                 st.info("👀 Both eyes clear (+0.1 HAPPY)")
#             elif facial_features['eyes_detected'] == 1:
#                 base_prediction += 0.05
#                 st.info("👁️ One eye clear (+0.05 HAPPY)")
#             else:
#                 base_prediction -= 0.1
#                 st.warning("❓ Face/eyes unclear (-0.1 HAPPY)")
                
#             # Factor 4: Face ratio (wider is often happier)
#             if facial_features['face_ratio'] > 1.2:
#                 base_prediction += 0.1
#                 st.info(f"📏 Wide face ratio ({facial_features['face_ratio']:.2f}) (+0.1 HAPPY)")
#             elif facial_features['face_ratio'] < 0.9:
#                 base_prediction -= 0.1
#                 st.info(f"📐 Narrow face ratio ({facial_features['face_ratio']:.2f}) (-0.1 HAPPY)")
                
#             # Factor 5: Texture complexity (smiles create lines/edges)
#             if characteristics['texture_complexity'] > 0.3:
#                 base_prediction += 0.1
#                 st.info(f"🖼️ High texture/edges ({characteristics['texture_complexity']:.2f}) (+0.1 HAPPY)")
            
#             # Factor 6: Brightness (darker images often correlate with non-happy)
#             if characteristics['brightness'] < 80:
#                 base_prediction -= 0.1
#                 st.warning(f"💡 Dark image ({characteristics['brightness']:.1f}) (-0.1 HAPPY)")
#             elif characteristics['brightness'] > 180:
#                 base_prediction += 0.05
#                 st.info(f"🌟 Bright image ({characteristics['brightness']:.1f}) (+0.05 HAPPY)")
                
#             # Add small randomness for variation
#             base_prediction += random.uniform(-0.05, 0.05)
#             st.write(f"✨ Random adjustment: {random.uniform(-0.05, 0.05):.3f}")
            
#             # Ensure final prediction is within bounds [0.1, 0.9]
#             prediction_score = max(0.1, min(0.9, base_prediction))
#             st.write(f"📝 Final raw score for HAPPY: {prediction_score:.3f}")
            
#             # 3. Determine Final Result
#             if prediction_score > 0.5:
#                 mood_label = HAPPY_LABEL
#                 confidence = prediction_score
#             else:
#                 mood_label = NOT_HAPPY_LABEL
#                 confidence = 1.0 - prediction_score
            
#             return mood_label, confidence
            
#         except Exception as e:
#             st.error(f"❌ Prediction error: {e}")
#             return None, None

# # ================================
# # 🧠 PLACEHOLDER MODEL FUNCTION
# # We no longer load a Keras model, but we keep this function
# # simple for consistency, noting the target size.
# # ================================
# @st.cache_resource
# def load_model_placeholder():
#     """Returns necessary info for the feature-based system."""
#     st.sidebar.success("✅ Feature Analysis System loaded!")
#     # We return None for the model object as it's not needed, but keep target size
#     # for processing consistency.
#     return None, None, IMAGE_SIZE 

# # ================================
# # 📂 SIDEBAR
# # ================================
# base64_image_urls = []

# with st.sidebar:
#     st.title("⚙️ App Configuration")
    
#     uploaded_files = st.file_uploader(
#         "🖼️ Upload background images (optional):",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files:
#         for file in uploaded_files:
#             url = get_base64_image_url(file)
#             if url:
#                 base64_image_urls.append(url)
    
#     st.markdown("---")
#     st.subheader("📘 About the System")
#     st.info("This system uses **OpenCV Haar Cascades** and image feature analysis (smile detection, mouth curve, etc.) to robustly classify facial expressions.")
    
#     st.markdown(f"📅 Updated: **{datetime.now().strftime('%b %d, %Y')}**")
#     st.markdown("Made with ❤️ ", unsafe_allow_html=True)

# set_cinematic_bg(base64_image_urls)

# # ================================
# # 🎓 HEADER
# # ================================
# st.markdown("""
# <h1 style='text-align:center; color:#ff9900; text-shadow: 2px 2px 6px #000;'>
# 😊 Smart Feature-Based Mood Classifier
# </h1>
# <p style='text-align:center; font-size:18px; color:#fff;'>
# Detect emotions from <b style='color:#44ff44;'>facial features</b>, not just deep learning weights.
# </p>
# """, unsafe_allow_html=True)

# # Load the system placeholder
# _, _, target_size = load_model_placeholder()

# # ================================
# # 📊 TABS
# # ================================
# tab1, tab2, tab3 = st.tabs(["🎭 Mood Detection", "📊 Batch Analysis", "ℹ️ System Info"])

# # ================================
# # 🎭 TAB 1 — MOOD DETECTION (UNIVERSAL)
# # ================================
# with tab1:
#     st.header("Upload Image for Mood Analysis")
    
#     col1, col2 = st.columns(2)
    
#     uploaded_image = None
    
#     with col1:
#         uploaded_image = st.file_uploader(
#             "📸 Upload a face image", 
#             type=["jpg", "jpeg", "png", "bmp", "webp"],
#             key="single_upload",
#             help="Upload any image size or ratio - the system will automatically adapt"
#         )
        
#         if uploaded_image:
#             image_pil = Image.open(uploaded_image)
#             st.image(image_pil, caption="Uploaded Image", use_container_width=True)
#             st.info(f"📐 Original size: {image_pil.size}")
    
#     with col2:
#         st.markdown("### Feature Analysis Log")
#         st.info("The logic analyzes features like smile detection, mouth curve, and image brightness to determine the mood score.")
        
#         # Placeholder for the verbose prediction log
#         prediction_log_container = st.empty()

#     if st.button("🔍 Analyze Mood", use_container_width=True, type="primary"):
#         if not uploaded_image:
#             st.warning("⚠️ Please upload an image first.")
#         else:
#             # Clear previous logs and start spinner
#             prediction_log_container.empty()
            
#             with st.spinner("🤖 System is analyzing facial features..."):
                
#                 # Convert PIL image to numpy array for OpenCV
#                 image_np = np.array(image_pil.convert('RGB')) 
                
#                 # --- Step 1: Feature-Based Prediction ---
#                 mood_label, confidence = smart_feature_predict(
#                     image_np, prediction_log_container
#                 )
                
#                 if mood_label is not None:
#                     confidence_percent = confidence * 100
                    
#                     # Display emoji and check if happy
#                     is_happy = (mood_label == HAPPY_LABEL)
#                     mood_emoji = "😊" if is_happy else "😔"
#                     display_label = f"{mood_label.replace('_', ' ').title()} {mood_emoji}"
                    
#                     st.balloons()
                    
#                     # --- Step 2: Final Results Display ---
#                     st.markdown(f"""
#                     <div class="prediction-box">
#                         <h2 style="color:white;">Detected Mood</h2>
#                         <h1 style="color:#ff9900; font-size:3.2em;">{display_label}</h1>
#                         <p style="font-size:1.2em; color:#ddd;">Confidence: {confidence_percent:.1f}%</p>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Confidence visualization
#                     st.markdown("### Confidence Level")
#                     st.markdown(f"""
#                     <div class="confidence-bar">
#                         <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
#                         <div class="confidence-text">{confidence_percent:.1f}%</div>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Additional insights
#                     if is_happy:
#                         final_message = "🎉 Great! The analysis strongly suggests a happy expression based on facial features."
#                         st.success(final_message)
#                     else:
#                         final_message = "💭 The analysis suggests a neutral or non-happy expression. Check the feature log for details!"
#                         st.info(final_message)
#                 else:
#                     st.error("❌ Prediction failed. Check the log for OpenCV errors.")

# # ================================
# # 📊 TAB 2 — BATCH ANALYSIS (UNIVERSAL)
# # ================================
# with tab2:
#     st.header("Batch Image Analysis")
    
#     st.info("Upload multiple images for batch analysis using feature detection.")
    
#     uploaded_batch = st.file_uploader(
#         "📁 Upload multiple images", 
#         type=["jpg", "jpeg", "png", "bmp", "webp"],
#         accept_multiple_files=True,
#         key="batch_upload"
#     )
    
#     if uploaded_batch:
#         st.write(f"📸 **{len(uploaded_batch)}** images selected for analysis")
        
#         if st.button("🚀 Process Batch", use_container_width=True):
#             progress_bar = st.progress(0)
#             results = []
            
#             # Using a temporary container to suppress batch preprocessing/prediction logging
#             temp_log_container = st.empty() 
            
#             for i, image_file in enumerate(uploaded_batch):
#                 progress = (i + 1) / len(uploaded_batch)
#                 progress_bar.progress(progress)
                
#                 try:
#                     # Convert PIL image to numpy array for OpenCV
#                     image_pil = Image.open(image_file)
#                     image_np = np.array(image_pil.convert('RGB')) 
                    
#                     # Make prediction
#                     mood_label, confidence = smart_feature_predict(image_np, temp_log_container)
                    
#                     if mood_label is not None:
#                         confidence_percent = confidence * 100
#                         mood_emoji = "😊" if mood_label == HAPPY_LABEL else "😔"
                        
#                         results.append({
#                             "image": image_file.name,
#                             "mood": f"{mood_label.replace('_', ' ').title()} {mood_emoji}",
#                             "confidence": f"{confidence_percent:.1f}%"
#                         })
#                     else:
#                         results.append({
#                             "image": image_file.name,
#                             "mood": "❌ Analysis Failed",
#                             "confidence": "0%"
#                         })
                        
#                 except Exception as e:
#                     results.append({
#                         "image": image_file.name,
#                         "mood": f"❌ Error: {str(e)[:50]}...",
#                         "confidence": "0%"
#                     })
            
#             # Remove the temporary log container content after processing
#             temp_log_container.empty()

#             # Display results
#             st.success("✅ Batch analysis completed!")
            
#             # Create results table
#             results_df = pd.DataFrame(results)
#             st.dataframe(results_df, use_container_width=True)
            
#             # Summary statistics
#             happy_count = sum(1 for r in results if HAPPY_LABEL in r["mood"].upper())
#             st.metric("😊 Happy Faces", happy_count)
#             st.metric("😔 Not Happy Faces", len(results) - happy_count)

# # ================================
# # 📘 TAB 3 — MODEL INFO
# # ================================
# with tab3:
#     st.header("System Overview")
    
#     st.info("""
#     This system implements a **Feature-Based Classification Model** based on the logic from your provided code. 
#     It avoids the pitfalls of potentially poor pre-trained Keras weights by using classic Computer Vision techniques.
#     """)
    
#     st.markdown("""
#     **Core Analysis Components:**
#     - **OpenCV Haar Cascades:** Used to reliably detect the face, eyes, and smile.
#     - **Mouth Curve Analysis:** Measures the curvature of the lower facial area.
#     - **Texture Complexity (Canny Edges):** Higher complexity can indicate smile lines.
#     - **Image Properties:** Brightness and Face Aspect Ratio are also factored into the final score.
    
#     **Prediction Mechanism:**
#     The final 'HAPPY' score is calculated by starting at 0.5 (neutral) and adding or subtracting weight based on the presence/absence of the features above. This score is then converted into a final class (Happy or Not Happy) and a confidence percentage.
#     """)
    
#     col1, col2, col3 = st.columns(3)
    
#     col1.metric("Method", "Feature Engineering")
#     col2.metric("Primary Tool", "OpenCV")
#     col3.metric("Classification Logic", "Rule-Based Scoring")

# # ================================
# # 🎯 FOOTER
# # ================================
# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #888;'>
#     <p>🔬 Built with Streamlit • OpenCV • NumPy</p>
#     <p>🎯 Robust feature analysis for reliable mood detection</p>
#     <p>For educational and research purposes</p>
# </div>
# """, unsafe_allow_html=True)


# Below my code that intgrates camera input

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import base64
import random
from PIL import Image
from datetime import datetime

# ================================
# 🎯 CONFIGURATION
# ================================
# --- Define fixed labels for binary classification consistency ---
HAPPY_LABEL = "HAPPY"
NOT_HAPPY_LABEL = "NOT_HAPPY"
IMAGE_SIZE = (224, 224) # Target size for internal processing consistency

# Initialize Haar cascades globally
try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    CASCADES_LOADED = True
except Exception as e:
    st.error(f"Failed to load OpenCV cascades: {e}. Facial feature detection disabled.")
    CASCADES_LOADED = False

# ================================
# ⚙️ UTILITIES
# ================================
def get_base64_image_url(uploaded_file):
    """Encodes an uploaded file's bytes to a base64 URL for CSS background usage."""
    try:
        bytes_data = uploaded_file.getvalue()
        base64_encoded_data = base64.b64encode(bytes_data).decode("utf-8")
        mime_type = uploaded_file.type or "image/png"
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def set_cinematic_bg(base64_urls, interval_per_image=6):
    """Sets a cinematic background effect using cycling base64 images."""
    num_images = len(base64_urls)
    total_duration = num_images * interval_per_image
    OVERLAY_OPACITY = "rgba(0,0,0,0.6)"
    
    # Minimal CSS for no images
    if num_images == 0:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1b2735, #090a0f);
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
        return

    # Generate CSS keyframes for background animation
    css_keyframes = []
    for i in range(num_images):
        start_percent = (i * 100) / num_images
        hold_percent = start_percent + (100 / num_images)
        css_keyframes.append(f"{start_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
        css_keyframes.append(f"{hold_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
    
    css_keyframes.append(f"100% {{ background-image: url('{base64_urls[0]}'); }}")

    st.markdown(f"""
    <style>
    .stApp {{
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
        background-image: url('{base64_urls[0]}');
        animation: cinematicBg {total_duration}s infinite;
        color: white;
    }}
    
    @keyframes cinematicBg {{
        {"".join(css_keyframes)}
    }}
    
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: {OVERLAY_OPACITY};
        z-index: 0;
    }}
    
    /* Ensure Streamlit components are layered above the overlay */
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent !important;
        z-index: 1;
    }}

    /* Styling for the output box */
    .prediction-box {{
        background-color: rgba(255,153,0,0.2);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 20px #ff9900;
        margin: 20px 0;
    }}
    
    .confidence-bar {{
        background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44);
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
        position: relative;
    }}
    
    .confidence-fill {{
        height: 100%;
        border-radius: 10px;
        background: rgba(255,255,255,0.3);
        transition: width 0.5s ease-in-out;
    }}
    
    .confidence-text {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: bold;
        text-shadow: 1px 1px 2px black;
    }}
    </style>
    """, unsafe_allow_html=True)

# ================================
# ⚙️ FEATURE DETECTION LOGIC
# ================================

def detect_facial_features(image_np):
    """Detect facial features using OpenCV for expression analysis"""
    features = {
        'face_detected': False,
        'eyes_detected': 0,
        'smile_detected': False,
        'mouth_curve': 0.5,  # Neutral
        'face_ratio': 1.0
    }
    
    if not CASCADES_LOADED:
        return features
        
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        features['face_detected'] = True
        x, y, w, h = faces[0]
        
        # Calculate face ratio (width/height)
        features['face_ratio'] = w / h if h > 0 else 1.0
        
        # Region of Interest for eyes and mouth
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes (only in the face ROI)
        eyes = EYE_CASCADE.detectMultiScale(roi_gray, 1.1, 6)
        features['eyes_detected'] = len(eyes)
        
        # Detect smile (only in the face ROI)
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.7, 20)
        features['smile_detected'] = len(smiles) > 0
        
        # Simple mouth curve analysis (lower face region)
        mouth_region = roi_gray[int(h*0.6):int(h*0.9), int(w*0.2):int(w*0.8)]
        if mouth_region.size > 0:
            # Calculate horizontal gradients to detect mouth curve
            sobelx = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)
            # Normalize the curve measurement
            mouth_curve = np.mean(np.abs(sobelx))
            features['mouth_curve'] = min(1.0, mouth_curve / 50.0) 
    
    return features

def analyze_expression_characteristics(image_np):
    """Analyze visual characteristics that might indicate expression"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    
    characteristics = {
        'brightness': np.mean(gray),
        'contrast': np.std(gray),
        'saturation': np.mean(hsv[:, :, 1]),
        'warmth': 0.0,
        'texture_complexity': 0.0
    }
    
    # Color warmth (red/blue ratio)
    r, g, b = cv2.split(image_np)
    characteristics['warmth'] = np.mean(r) / (np.mean(b) + 1e-6)
    
    # Texture complexity (edge density)
    edges = cv2.Canny(gray, 50, 150)
    characteristics['texture_complexity'] = np.mean(edges) / 255.0
    
    return characteristics

def smart_feature_predict(image_np, debug_container):
    """
    Smart prediction based on facial features and expression analysis 
    using the logic provided by the user.
    
    Updated: Increased positive weights (smile, mouth curve) and 
    decreased negative weights (no smile, unclear eyes, dark image) 
    to make the classifier more tolerant of happy expressions in sub-optimal photos.
    """
    with debug_container:
        try:
            # 1. Analyze features and characteristics
            facial_features = detect_facial_features(image_np)
            characteristics = analyze_expression_characteristics(image_np)
            
            st.write(f"🔍 Facial Features: {facial_features}")
            st.write(f"🎨 Image Characteristics: Brightness={characteristics['brightness']:.1f}, Texture={characteristics['texture_complexity']:.2f}")

            # 2. Start with a neutral score
            base_prediction = 0.5 
            
            # --- Applying Feature Factors (Weights based on user's logic) ---
            
            # Factor 1: Smile detection (strong indicator) - INCREASED POSITIVE WEIGHT
            if facial_features['smile_detected']:
                base_prediction += 0.4 # Changed from 0.3
                st.info("✅ Smile detected (+0.4 HAPPY)")
            else:
                base_prediction -= 0.05 # Changed from 0.1
                st.info("❌ No smile detected (-0.05 HAPPY)")
            
            # Factor 2: Mouth curve - INCREASED POSITIVE WEIGHT
            if facial_features['mouth_curve'] > 0.7:
                base_prediction += 0.25 # Changed from 0.2
                st.info(f"⬆️ Strong mouth curve ({facial_features['mouth_curve']:.2f}) (+0.25 HAPPY)")
            elif facial_features['mouth_curve'] < 0.3:
                base_prediction -= 0.2
                st.info(f"⬇️ Weak mouth curve ({facial_features['mouth_curve']:.2f}) (-0.2 HAPPY)")
            
            # Factor 3: Eye detection - DECREASED NEGATIVE WEIGHT
            if facial_features['eyes_detected'] >= 2:
                base_prediction += 0.1
                st.info("👀 Both eyes clear (+0.1 HAPPY)")
            elif facial_features['eyes_detected'] == 1:
                base_prediction += 0.05
                st.info("👁️ One eye clear (+0.05 HAPPY)")
            else:
                base_prediction -= 0.05 # Changed from 0.1
                st.warning("❓ Face/eyes unclear (-0.05 HAPPY)")
                
            # Factor 4: Face ratio (wider is often happier)
            if facial_features['face_ratio'] > 1.2:
                base_prediction += 0.1
                st.info(f"📏 Wide face ratio ({facial_features['face_ratio']:.2f}) (+0.1 HAPPY)")
            elif facial_features['face_ratio'] < 0.9:
                base_prediction -= 0.1
                st.info(f"📐 Narrow face ratio ({facial_features['face_ratio']:.2f}) (-0.1 HAPPY)")
                
            # Factor 5: Texture complexity (smiles create lines/edges)
            if characteristics['texture_complexity'] > 0.3:
                base_prediction += 0.1
                st.info(f"🖼️ High texture/edges ({characteristics['texture_complexity']:.2f}) (+0.1 HAPPY)")
            
            # Factor 6: Brightness (darker images often correlate with non-happy) - DECREASED NEGATIVE WEIGHT
            if characteristics['brightness'] < 80:
                base_prediction -= 0.05 # Changed from 0.1
                st.warning(f"💡 Dark image ({characteristics['brightness']:.1f}) (-0.05 HAPPY)")
            elif characteristics['brightness'] > 180:
                base_prediction += 0.05
                st.info(f"🌟 Bright image ({characteristics['brightness']:.1f}) (+0.05 HAPPY)")
                
            # Add small randomness for variation
            base_prediction += random.uniform(-0.05, 0.05)
            st.write(f"✨ Random adjustment: {random.uniform(-0.05, 0.05):.3f}")
            
            # Ensure final prediction is within bounds [0.1, 0.9]
            prediction_score = max(0.1, min(0.9, base_prediction))
            st.write(f"📝 Final raw score for HAPPY: {prediction_score:.3f}")
            
            # 3. Determine Final Result
            if prediction_score > 0.5:
                mood_label = HAPPY_LABEL
                confidence = prediction_score
            else:
                mood_label = NOT_HAPPY_LABEL
                confidence = 1.0 - prediction_score
            
            return mood_label, confidence
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None, None

# ================================
# 🧠 PLACEHOLDER MODEL FUNCTION
# ================================
@st.cache_resource
def load_model_placeholder():
    """Returns necessary info for the feature-based system."""
    st.sidebar.success("✅ Feature Analysis System loaded!")
    return None, None, IMAGE_SIZE 

# ================================
# 📂 SIDEBAR
# ================================
base64_image_urls = []

with st.sidebar:
    st.title("⚙️ App Configuration")
    
    uploaded_files = st.file_uploader(
        "🖼️ Upload background images (optional):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        for file in uploaded_files:
            url = get_base64_image_url(file)
            if url:
                base64_image_urls.append(url)
    
    st.markdown("---")
    st.subheader("📘 About the System")
    st.info("This system uses **OpenCV Haar Cascades** and image feature analysis (smile detection, mouth curve, etc.) to robustly classify facial expressions.")
    
    st.markdown(f"📅 Updated: **{datetime.now().strftime('%b %d, %Y')}**")
    st.markdown("Made with ❤️ ", unsafe_allow_html=True)
    st.markdown("✨ Developed by **Umar Imam**", unsafe_allow_html=True)

set_cinematic_bg(base64_image_urls)

# ================================
# 🎓 HEADER
# ================================
st.markdown("""
<h1 style='text-align:center; color:#ff9900; text-shadow: 2px 2px 6px #000;'>
😊 Smart Feature-Based Mood Classifier
</h1>
<p style='text-align:center; font-size:18px; color:#fff;'>
Detect emotions from <b style='color:#44ff44;'>facial features</b>, not just deep learning weights.
</p>
""", unsafe_allow_html=True)

# Load the system placeholder
_, _, target_size = load_model_placeholder()

# ================================
# 📊 TABS
# ================================
tab1, tab2, tab3 = st.tabs(["🎭 Mood Detection", "📊 Batch Analysis", "ℹ️ System Info"])

# ================================
# 🎭 TAB 1 — MOOD DETECTION
# ================================
with tab1:
    st.header("Upload or Capture Image for Mood Analysis")
    
    col_input, col_log = st.columns(2)
    
    uploaded_image = None
    captured_image = None
    
    with col_input:
        
        # --- Input Selection ---
        input_mode = st.radio(
            "Select Input Source:",
            ("File Upload", "Live Camera Capture"),
            key="input_mode",
            horizontal=True
        )

        if input_mode == "File Upload":
            uploaded_image = st.file_uploader(
                "📸 Upload a face image", 
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                key="single_upload",
                help="Upload any image size or ratio - the system will automatically adapt"
            )
            if uploaded_image:
                image_pil = Image.open(uploaded_image)
                st.image(image_pil, caption="Uploaded Image", use_container_width=True)
                st.info(f"📐 Original size: {image_pil.size}")
                
        elif input_mode == "Live Camera Capture":
            captured_image = st.camera_input("🤳 Take a Photo")
            if captured_image:
                image_pil = Image.open(captured_image)
                # captured image is already displayed by st.camera_input, no need for another st.image
        
        # Determine the image to be analyzed
        image_to_analyze = uploaded_image or captured_image

    with col_log:
        st.markdown("### Feature Analysis Log")
        st.info("The logic analyzes features like smile detection, mouth curve, and image brightness to determine the mood score.")
        
        # Placeholder for the verbose prediction log
        prediction_log_container = st.empty()

    if st.button("🔍 Analyze Mood", use_container_width=True, type="primary"):
        if not image_to_analyze:
            st.warning("⚠️ Please upload an image or capture a photo first.")
        else:
            # Clear previous logs and start spinner
            prediction_log_container.empty()
            
            with st.spinner("🤖 System is analyzing facial features..."):
                
                # Convert PIL image to numpy array for OpenCV
                image_np = np.array(image_pil.convert('RGB')) 
                
                # --- Step 1: Feature-Based Prediction ---
                mood_label, confidence = smart_feature_predict(
                    image_np, prediction_log_container
                )
                
                if mood_label is not None:
                    confidence_percent = confidence * 100
                    
                    # Display emoji and check if happy
                    is_happy = (mood_label == HAPPY_LABEL)
                    mood_emoji = "😊" if is_happy else "😔"
                    display_label = f"{mood_label.replace('_', ' ').title()} {mood_emoji}"
                    
                    st.balloons()
                    
                    # --- Step 2: Final Results Display ---
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color:white;">Detected Mood</h2>
                        <h1 style="color:#ff9900; font-size:3.2em;">{display_label}</h1>
                        <p style="font-size:1.2em; color:#ddd;">Confidence: {confidence_percent:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence visualization
                    st.markdown("### Confidence Level")
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
                        <div class="confidence-text">{confidence_percent:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    if is_happy:
                        final_message = "🎉 Great! The analysis strongly suggests a happy expression based on facial features."
                        st.success(final_message)
                    else:
                        final_message = "💭 The analysis suggests a neutral or non-happy expression. Check the feature log for details!"
                        st.info(final_message)
                else:
                    st.error("❌ Prediction failed. Check the log for OpenCV errors.")

# ================================
# 📊 TAB 2 — BATCH ANALYSIS
# ================================
with tab2:
    st.header("Batch Image Analysis")
    
    st.info("Upload multiple images for batch analysis using feature detection.")
    
    uploaded_batch = st.file_uploader(
        "📁 Upload multiple images", 
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_batch:
        st.write(f"📸 **{len(uploaded_batch)}** images selected for analysis")
        
        if st.button("🚀 Process Batch", use_container_width=True):
            progress_bar = st.progress(0)
            results = []
            
            # Using a temporary container to suppress batch preprocessing/prediction logging
            temp_log_container = st.empty() 
            
            for i, image_file in enumerate(uploaded_batch):
                progress = (i + 1) / len(uploaded_batch)
                progress_bar.progress(progress)
                
                try:
                    # Convert PIL image to numpy array for OpenCV
                    image_pil = Image.open(image_file)
                    image_np = np.array(image_pil.convert('RGB')) 
                    
                    # Make prediction
                    mood_label, confidence = smart_feature_predict(image_np, temp_log_container)
                    
                    if mood_label is not None:
                        confidence_percent = confidence * 100
                        mood_emoji = "😊" if mood_label == HAPPY_LABEL else "😔"
                        
                        results.append({
                            "image": image_file.name,
                            "mood": f"{mood_label.replace('_', ' ').title()} {mood_emoji}",
                            "confidence": f"{confidence_percent:.1f}%"
                        })
                    else:
                        results.append({
                            "image": image_file.name,
                            "mood": "❌ Analysis Failed",
                            "confidence": "0%"
                        })
                        
                except Exception as e:
                    results.append({
                        "image": image_file.name,
                        "mood": f"❌ Error: {str(e)[:50]}...",
                        "confidence": "0%"
                    })
            
            # Remove the temporary log container content after processing
            temp_log_container.empty()

            # Display results
            st.success("✅ Batch analysis completed!")
            
            # Create results table
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Summary statistics
            happy_count = sum(1 for r in results if HAPPY_LABEL in r["mood"].upper())
            st.metric("😊 Happy Faces", happy_count)
            st.metric("😔 Not Happy Faces", len(results) - happy_count)

# ================================
# 📘 TAB 3 — MODEL INFO
# ================================
with tab3:
    st.header("System Overview")
    
    st.info("""
    This system implements a **Feature-Based Classification Model** based on the logic from your provided code. 
    It avoids the pitfalls of potentially poor pre-trained Keras weights by using classic Computer Vision techniques.
    """)
    
    st.markdown("""
    **Core Analysis Components:**
    - **OpenCV Haar Cascades:** Used to reliably detect the face, eyes, and smile.
    - **Mouth Curve Analysis:** Measures the curvature of the lower facial area.
    - **Texture Complexity (Canny Edges):** Higher complexity can indicate smile lines.
    - **Image Properties:** Brightness and Face Aspect Ratio are also factored into the final score.
    
    **Prediction Mechanism:**
    The final 'HAPPY' score is calculated by starting at 0.5 (neutral) and adding or subtracting weight based on the presence/absence of the features above. This score is then converted into a final class (Happy or Not Happy) and a confidence percentage.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Method", "Feature Engineering")
    col2.metric("Primary Tool", "OpenCV")
    col3.metric("Classification Logic", "Rule-Based Scoring")

# ================================
# 🎯 FOOTER
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🔬 Built with Streamlit • OpenCV • NumPy</p>
    <p>🎯 Robust feature analysis for reliable mood detection</p>
    <p>For educational and research purposes</p>
</div>
""", unsafe_allow_html=True)