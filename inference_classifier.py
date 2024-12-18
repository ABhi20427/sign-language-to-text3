import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import math

# Load pre-trained model
MODEL_PATH = './models/sign_language_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No trained model found. Please run train_classifier.py first.")

# Load model and labels
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    labels = model_data.get('labels', [])

# Create label dictionary (adjust as needed)
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z mapping

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=2
)

def extract_hand_features(hand_landmarks, frame_width, frame_height):
    """
    Extract consistent 84-feature vector for hand landmarks
    """
    # Extract landmark coordinates
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]

    # Normalization parameters
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Feature vector to store detailed hand information
    feature_vector = []

    for i, landmark in enumerate(hand_landmarks.landmark):
        # Normalized coordinates
        x_norm = (landmark.x - x_min) / (x_max - x_min)
        y_norm = (landmark.y - y_min) / (y_max - y_min)

        # Add normalized coordinates
        feature_vector.append(x_norm)
        feature_vector.append(y_norm)

        # Add angle and distance features if possible
        if i > 0:
            prev_landmark = hand_landmarks.landmark[i-1]
            
            # Angle between consecutive landmarks
            angle = math.atan2(
                landmark.y - prev_landmark.y, 
                landmark.x - prev_landmark.x
            )
            
            # Distance between landmarks
            distance = math.sqrt(
                (landmark.x - prev_landmark.x)**2 + 
                (landmark.y - prev_landmark.y)**2
            )
            
            feature_vector.append(angle)
            feature_vector.append(distance)

    # Ensure exactly 84 features (21 landmarks * 4 features)
    if len(feature_vector) > 84:
        feature_vector = feature_vector[:84]
    elif len(feature_vector) < 84:
        # Pad with zeros
        feature_vector.extend([0.0] * (84 - len(feature_vector)))

    return feature_vector

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Prediction stabilization
    prediction_buffer = []
    BUFFER_SIZE = 5
    CONFIDENCE_THRESHOLD = 0.7

    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip frame for mirror-like experience
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = hands.process(frame_rgb)

        # Reset prediction buffer if no hands detected
        if not results.multi_hand_landmarks:
            prediction_buffer.clear()

        # Process each detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract enhanced features
                try:
                    # Extract features with exactly 84 dimensions
                    features = extract_hand_features(
                        hand_landmarks, frame.shape[1], frame.shape[0]
                    )

                    # Reshape and predict
                    features_array = np.array(features).reshape(1, -1)
                    
                    # Predict sign
                    prediction = model.predict(features_array)
                    predicted_character = labels_dict[int(prediction[0])]

                    # Stabilize prediction
                    prediction_buffer.append(predicted_character)
                    if len(prediction_buffer) > BUFFER_SIZE:
                        prediction_buffer.pop(0)

                    # Get most common prediction
                    if len(prediction_buffer) == BUFFER_SIZE:
                        from collections import Counter
                        most_common = Counter(prediction_buffer).most_common(1)[0]
                        if most_common[1] / BUFFER_SIZE >= CONFIDENCE_THRESHOLD:
                            final_prediction = most_common[0]
                        else:
                            final_prediction = predicted_character
                    else:
                        final_prediction = predicted_character

                    # Draw bounding box and prediction
                    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                    
                    x1 = int(min(x_coords) * frame.shape[1])
                    y1 = int(min(y_coords) * frame.shape[0])
                    x2 = int(max(x_coords) * frame.shape[1])
                    y2 = int(max(y_coords) * frame.shape[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, final_prediction, 
                                (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error processing landmarks: {e}")

        # Display frame
        cv2.imshow('Sign Language Detection', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()