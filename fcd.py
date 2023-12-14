import streamlit as st
import cv2
import numpy as np

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Include external CSS using an HTML template

# st.markdown(open("styles.html").read(), unsafe_allow_html=True)

def detect_faces(uploaded_image):
    try:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Check if the image has 3 channels (BGR format)
        if image.shape[2] != 3:
            st.error("Uploaded image should have 3 color channels (BGR format).")
            return

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image

    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.title("Facial Recognition App")
    st.markdown("Upload an image to detect faces.")

    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        result_image = detect_faces(uploaded_image)
        if result_image is not None:
            st.image(result_image, channels="BGR", caption="Image with Detected Faces", use_column_width=True)

if __name__ == "__main__":
    main()
