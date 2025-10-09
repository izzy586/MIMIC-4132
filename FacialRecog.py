import cv2
import face_recognition
import os
import numpy as np

# Create a directory to store training images if it doesn't exist
os.makedirs('faces', exist_ok=True)

# Get the name for encoding
name = input("Enter name: ")

# Initialize video capture
cap = cv2.VideoCapture(0)

print("Camera ready! Press 'c' to capture, 'q' to quit")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break
    
    # Display the captured frame
    cv2.imshow("Training - Press 'c' to capture or 'q' to quit", frame)
    
    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    
    # Capture the frame when 'c' is pressed
    if key == ord('c') or key == ord('C'):
        print("Capturing image...")
        img_path = f'faces/{name}.jpg'
        cv2.imwrite(img_path, frame)  # Save the original frame
        print(f"Image saved at: {img_path}")
        
        # Convert the frame to RGB and find encodings
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img_rgb)
        
        if encodings:
            # Save the face encoding as a numpy array
            np.save(f'faces/{name}_encoding.npy', encodings[0])
            print(f"Encoding saved for {name}")
            print("Press 'q' to quit or capture another person")
        else:
            print("No face detected. Try again.")
    
    # Exit the loop when 'q' is pressed
    elif key == ord('q') or key == ord('Q'):
        print("Exiting...")
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()