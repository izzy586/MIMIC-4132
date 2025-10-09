import os
import time
import json
import serial
import pyttsx3
import random
import threading
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Use Ollama instead of OpenAI
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] Ollama not installed. Install with: pip3 install ollama")

# Optional: Face recognition (can work without it)
FACE_RECOGNITION_AVAILABLE = False
try:
    import cv2
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    print(f"[INFO] Face recognition not available: {e}")
    print("[INFO] Robot will work without face recognition")
except Exception as e:
    print(f"[INFO] Face recognition disabled due to error: {e}")
    print("[INFO] Robot will work without face recognition")

# Load environment variables
load_dotenv()

# ------------------------
# CONFIG VARIABLES 
# ------------------------
# Serial configuration
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
BAUD_RATE = int(os.getenv("BAUD_RATE", "115200"))
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "false").lower() == "true"

# Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Servo configuration
SERVO_MIN = 0
SERVO_MAX = 180
STEP_DELAY = 0.05
INTERPOLATION_STEPS = 20
DEFAULT_TTS_RATE = 150
DEFAULT_TTS_VOLUME = 1.0

# Face recognition settings
FACE_RECOGNITION_ENABLED = True
FACE_CHECK_INTERVAL = 1.0
GREETING_COOLDOWN = 30

# Servo pins
SERVOS = {
    "left_arm": 3,
    "right_arm": 5,
    "head": 6,
    "jaw": 9,
    "eyebrow_left": 10,
    "eyebrow_right": 11,
    "mouth_left": 12,
    "mouth_right": 13
}

# Predefined Movements
MOVEMENTS = {
    "wave": {
        "left_arm": [90, 0, 90, 0],
        "right_arm": [90, 90, 90, 90],
        "head": [90, 80, 100, 90],
        "jaw": [0],
        "speech": "Hello, I am waving!"
    },
    "nod": {
        "head": [90, 60, 120, 90],
        "speech": "I am nodding!"
    },
    "open_mouth": {
        "jaw": [0, 45, 0],
        "speech": "Watch me open my mouth!"
    },
    "look_around": {
        "head": [90, 60, 120, 90, 100, 80],
        "speech": "Looking around..."
    },
    "eyebrow_raise": {
        "eyebrow_left": [0, 45, 0],
        "eyebrow_right": [0, 45, 0],
        "speech": "Eyebrows up!"
    },
    "greet_friend": {
        "left_arm": [90, 0, 90],
        "right_arm": [90, 180, 90],
        "head": [90, 80, 100, 90],
        "jaw": [0, 30, 0],
        "speech": "Hey! Great to see you!"
    },
    "excited": {
        "left_arm": [90, 45, 135, 90],
        "right_arm": [90, 135, 45, 90],
        "head": [90, 70, 110, 90],
        "eyebrow_left": [0, 45, 0],
        "eyebrow_right": [0, 45, 0],
        "speech": "I'm so excited to see you!"
    }
}

SEQUENCES_DIR = "sequences"
os.makedirs(SEQUENCES_DIR, exist_ok=True)

# Global variables for face recognition
face_recognition_active = False
last_seen = {}
known_encodings = []
class_names = []

# ------------------------
# DEBUG LOGGING (moved up before TTS init)
# ------------------------
def log_event(event_type, message):
    """Log events with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{event_type}] {message}")

# ------------------------
# INITIALIZE TTS
# ------------------------
def initialize_tts():
    """Initialize TTS engine with error handling"""
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", DEFAULT_TTS_RATE)
        engine.setProperty("volume", DEFAULT_TTS_VOLUME)
        log_event("INFO", "TTS engine initialized successfully")
        return engine
    except Exception as e:
        log_event("ERROR", f"Failed to initialize TTS engine: {e}")
        return None

tts_engine = initialize_tts()

# ------------------------
# FACE RECOGNITION FUNCTIONS
# ------------------------
def load_face_encodings(encodings_path='faces'):
    """Load face encodings from saved files"""
    global known_encodings, class_names
    encodings = []
    names = []
    
    try:
        if not os.path.exists(encodings_path):
            log_event("WARNING", f"Face encodings directory '{encodings_path}' not found")
            return [], []
        
        for file in os.listdir(encodings_path):
            if file.endswith("_encoding.npy"):
                name = file.split('_')[0]
                encoding = np.load(os.path.join(encodings_path, file))
                encodings.append(encoding)
                names.append(name)
        
        log_event("INFO", f"Loaded {len(names)} face encodings: {names}")
        return encodings, names
    except Exception as e:
        log_event("ERROR", f"Failed to load face encodings: {e}")
        return [], []

def recognize_face(frame):
    """Detect and recognize faces in frame"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = class_names[first_match_index]
            
            recognized_names.append(name)
        
        return recognized_names
    except Exception as e:
        log_event("ERROR", f"Face recognition error: {e}")
        return []

def face_recognition_loop(ser):
    """Background thread for continuous face recognition"""
    global face_recognition_active, last_seen
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log_event("ERROR", "Failed to open camera for face recognition")
        return
    
    log_event("INFO", "Face recognition started")
    
    while face_recognition_active:
        try:
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
            
            names = recognize_face(frame)
            current_time = time.time()
            
            for name in names:
                if name == "Unknown":
                    continue
                
                last_greeting = last_seen.get(name, 0)
                if current_time - last_greeting > GREETING_COOLDOWN:
                    log_event("INFO", f"Recognized: {name}")
                    greet_person(ser, name)
                    last_seen[name] = current_time
            
            time.sleep(FACE_CHECK_INTERVAL)
        
        except Exception as e:
            log_event("ERROR", f"Error in face recognition loop: {e}")
            time.sleep(1)
    
    cap.release()
    log_event("INFO", "Face recognition stopped")

def greet_person(ser, name):
    """Greet a recognized person"""
    display_name = "Izzy" if name.lower() == "izyan" else name
    
    greetings = [
        f"Hello {display_name}! Nice to see you!",
        f"Hey {display_name}! How are you doing?",
        f"Hi {display_name}! Welcome back!",
        f"{display_name}! Great to see you again!"
    ]
    
    greeting = random.choice(greetings)
    movement = MOVEMENTS["greet_friend"].copy()
    movement["speech"] = greeting
    
    speech = movement.pop("speech", None)
    execute_movement(ser, movement)
    if speech:
        speak_text(speech)

def start_face_recognition(ser):
    """Start face recognition in background thread"""
    global face_recognition_active, known_encodings, class_names
    
    if not FACE_RECOGNITION_AVAILABLE:
        log_event("WARNING", "Face recognition libraries not installed")
        return
    
    if not FACE_RECOGNITION_ENABLED:
        log_event("WARNING", "Face recognition is disabled")
        return
    
    known_encodings, class_names = load_face_encodings()
    if not known_encodings:
        log_event("WARNING", "No face encodings loaded. Train faces first!")
        return
    
    face_recognition_active = True
    thread = threading.Thread(target=face_recognition_loop, args=(ser,), daemon=True)
    thread.start()
    log_event("INFO", "Face recognition thread started")

def stop_face_recognition():
    """Stop face recognition"""
    global face_recognition_active
    face_recognition_active = False
    log_event("INFO", "Stopping face recognition...")

# ------------------------
# SERIAL FUNCTIONS
# ------------------------
def connect_serial(port, baud):
    """Connect to serial port with error handling"""
    if SIMULATION_MODE:
        log_event("INFO", "Running in SIMULATION MODE - no serial connection")
        return "SIMULATION"
    
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        log_event("INFO", f"Connected to {port} at {baud} baud")
        return ser
    except serial.SerialException as e:
        log_event("ERROR", f"Serial connection failed: {e}")
        return None
    except Exception as e:
        log_event("ERROR", f"Unexpected error connecting to serial: {e}")
        return None

def send_servo_command(ser, servo_name, position):
    """Send servo command with error handling"""
    if ser == "SIMULATION":
        print(f"[SIMULATION] Servo {servo_name} -> {position}°")
        return True
    
    if not ser or not ser.is_open:
        log_event("ERROR", "Serial connection not available")
        return False
    
    try:
        position = max(SERVO_MIN, min(SERVO_MAX, int(position)))
        command = f"{servo_name}:{position}\n"
        ser.write(command.encode())
        time.sleep(STEP_DELAY)
        return True
    except serial.SerialException as e:
        log_event("ERROR", f"Failed to send servo command: {e}")
        return False

# ------------------------
# MOVEMENT UTILITIES
# ------------------------
def interpolate_positions(start, end, steps=INTERPOLATION_STEPS):
    """Generate interpolated positions between start and end"""
    if steps <= 1:
        yield end
        return
    for i in range(steps):
        yield int(start + (end - start) * (i / (steps - 1)))

def execute_movement(ser, movement):
    """Execute servo movement sequence"""
    if not movement:
        log_event("WARNING", "No movement data provided")
        return
    
    try:
        servo_movements = {k: v for k, v in movement.items() if k != "speech" and isinstance(v, list)}
        if not servo_movements:
            log_event("WARNING", "No valid servo movements in sequence")
            return
        
        max_steps = max(len(positions) for positions in servo_movements.values())
        previous_positions = {servo: positions[0] for servo, positions in servo_movements.items()}

        for step in range(max_steps):
            for servo, positions in servo_movements.items():
                if servo not in SERVOS:
                    log_event("WARNING", f"Unknown servo: {servo}")
                    continue
                
                target = positions[step % len(positions)]
                for pos in interpolate_positions(previous_positions.get(servo, target), target):
                    if not send_servo_command(ser, servo, pos):
                        return
                previous_positions[servo] = target
            time.sleep(0.1)
        
        log_event("INFO", "Movement sequence completed")
    except Exception as e:
        log_event("ERROR", f"Error executing movement: {e}")

# ------------------------
# TTS FUNCTION
# ------------------------
def speak_text(text):
    """Speak text using TTS"""
    print(f"[TTS] {text}")
    
    try:
        # Use subprocess to run TTS in separate process (more reliable on Mac)
        import subprocess
        import sys
        
        # Create a simple Python command to speak
        tts_command = f"""
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', {DEFAULT_TTS_RATE})
engine.setProperty('volume', {DEFAULT_TTS_VOLUME})
engine.say({repr(text)})
engine.runAndWait()
"""
        
        subprocess.run([sys.executable, '-c', tts_command], check=True)
        
    except Exception as e:
        log_event("ERROR", f"TTS failed: {e}")
        import traceback
        traceback.print_exc()

# ------------------------
# OLLAMA INTEGRATION
# ------------------------
def ask_llm(prompt):
    """Query local Ollama LLM for movement generation"""
    if not OLLAMA_AVAILABLE:
        log_event("ERROR", "Ollama not available. Install with: pip3 install ollama")
        return None
    
    system_message = """You are an assistant that generates safe servo movement sequences for a 3D-printed animatronic AND conversational responses.
Return ONLY a valid JSON object with:
- servo names as keys with lists of positions (0-180 degrees) as values
- a 'speech' key with a natural, conversational response (1-2 sentences)

Available servos: left_arm, right_arm, head, jaw, eyebrow_left, eyebrow_right, mouth_left, mouth_right.
Keep movements smooth, expressive, and safe. Use 2-5 positions per servo.
Match the movement to the emotion/content of your response.

Examples:
- Happy response: raise eyebrows, tilt head slightly, maybe raise arms
- Thinking: tilt head, slight eyebrow movement
- Excited: more dramatic arm and head movements
- Neutral chat: subtle head nods or tilts"""
    
    try:
        log_event("INFO", f"Asking Ollama ({OLLAMA_MODEL})...")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        answer_text = response['message']['content'].strip()
        
        # Extract JSON from response
        if "```json" in answer_text:
            answer_text = answer_text.split("```json")[1].split("```")[0].strip()
        elif "```" in answer_text:
            answer_text = answer_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(answer_text)
    except json.JSONDecodeError as e:
        log_event("ERROR", f"Ollama returned invalid JSON: {e}")
        print(f"[DEBUG] Raw response: {answer_text}")
        return None
    except Exception as e:
        log_event("ERROR", f"Ollama request failed: {e}")
        log_event("INFO", "Make sure Ollama is running: ollama serve")
        return None

def execute_llm_movement(ser, prompt):
    """Generate and execute LLM-based movement with conversational response"""
    log_event("INFO", f"Requesting AI response for: {prompt}")
    movement = ask_llm(prompt)
    
    if movement:
        speech = movement.pop("speech", None)
        execute_movement(ser, movement)
        if speech:
            speak_text(speech)
        return movement
    return None

# ------------------------
# SEQUENCE SAVE/LOAD
# ------------------------
def save_sequence(name, movement):
    """Save movement sequence to file"""
    try:
        filename = os.path.join(SEQUENCES_DIR, f"{name}.json")
        with open(filename, "w") as f:
            json.dump(movement, f, indent=4)
        log_event("INFO", f"Sequence '{name}' saved to {filename}")
    except Exception as e:
        log_event("ERROR", f"Failed to save sequence: {e}")

def load_sequence(name):
    """Load movement sequence from file"""
    try:
        filename = os.path.join(SEQUENCES_DIR, f"{name}.json")
        if not os.path.exists(filename):
            log_event("ERROR", f"Sequence '{name}' not found")
            return None
        with open(filename, "r") as f:
            movement = json.load(f)
        log_event("INFO", f"Sequence '{name}' loaded")
        return movement
    except Exception as e:
        log_event("ERROR", f"Failed to load sequence: {e}")
        return None

def list_sequences():
    """List all saved sequences"""
    try:
        sequences = [f.replace(".json", "") for f in os.listdir(SEQUENCES_DIR) if f.endswith(".json")]
        if sequences:
            print("\n[INFO] Saved sequences:")
            for seq in sequences:
                print(f"  - {seq}")
        else:
            print("[INFO] No saved sequences found")
    except Exception as e:
        log_event("ERROR", f"Failed to list sequences: {e}")

# ------------------------
# RANDOM MOVEMENTS
# ------------------------
def random_servo_movement():
    """Generate random servo movements"""
    movement = {}
    for servo in SERVOS.keys():
        movement[servo] = [random.randint(SERVO_MIN, SERVO_MAX) for _ in range(random.randint(2, 5))]
    movement["speech"] = "Random movement activated"
    return movement

# ------------------------
# MAIN INTERACTIVE LOOP
# ------------------------
def main():
    """Main program loop"""
    ser = connect_serial(SERIAL_PORT, BAUD_RATE)
    if not ser:
        print("[ERROR] Failed to connect to serial port.")
        print("[INFO] To run without Arduino, add SIMULATION_MODE=true to your .env file")
        return
    
    last_movement = None

    print("\n" + "="*60)
    print("  Mimic Animatronic Controller with Local AI (Ollama)")
    print("="*60)
    
    if not OLLAMA_AVAILABLE:
        print("\n[WARNING] Ollama not installed!")
        print("Install: pip3 install ollama")
        print("Then run: ollama serve (in another terminal)")
        print(f"And: ollama pull {OLLAMA_MODEL}\n")
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("\n[INFO] Face recognition disabled (libraries not installed)")
        enable_face_recognition = False
    else:
        # Ask user if they want face recognition
        while True:
            face_choice = input("\nEnable face recognition? (YES/NO): ").strip().upper()
            if face_choice in ["YES", "NO"]:
                break
            print("[WARNING] Please enter YES or NO")
        enable_face_recognition = (face_choice == "YES")
    
    print("\nCommands:")
    print("  - Predefined: wave, nod, open_mouth, look_around, eyebrow_raise")
    print("  - Face Recognition:")
    print("    * start_face: Enable face recognition")
    print("    * stop_face: Disable face recognition")
    print("  - AI Chat: Type anything - robot responds with speech + movement")
    print("  - save <n>: Save last movement")
    print("  - load <n>: Load saved movement")
    print("  - list: Show all saved sequences")
    print("  - random: Random movement")
    print("  - exit: Quit program")
    print("="*60 + "\n")

    # Start face recognition if user chose YES
    if enable_face_recognition and FACE_RECOGNITION_ENABLED:
        start_face_recognition(ser)

    try:
        while True:
            try:
                user_input = input("Command: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    log_event("INFO", "Exiting program")
                    stop_face_recognition()
                    break
                
                elif user_input.lower() == "start_face":
                    start_face_recognition(ser)
                
                elif user_input.lower() == "stop_face":
                    stop_face_recognition()
                
                elif user_input.lower() == "list":
                    list_sequences()
                
                elif user_input.startswith("save "):
                    name = user_input[5:].strip()
                    if not name:
                        log_event("WARNING", "Please provide a sequence name")
                    elif last_movement:
                        save_sequence(name, last_movement)
                    else:
                        log_event("WARNING", "No movement to save")
                
                elif user_input.startswith("load "):
                    name = user_input[5:].strip()
                    if not name:
                        log_event("WARNING", "Please provide a sequence name")
                    else:
                        movement = load_sequence(name)
                        if movement:
                            speech = movement.pop("speech", None)
                            execute_movement(ser, movement)
                            if speech:
                                speak_text(speech)
                            last_movement = movement
                
                elif user_input.lower() == "random":
                    movement = random_servo_movement()
                    execute_movement(ser, movement)
                    speak_text(movement.get("speech", "Random movement"))
                    last_movement = movement
                
                elif user_input in MOVEMENTS:
                    movement = MOVEMENTS[user_input].copy()
                    speech = movement.pop("speech", None)
                    execute_movement(ser, movement)
                    if speech:
                        speak_text(speech)
                    last_movement = MOVEMENTS[user_input]
                
                else:
                    # AI conversational response with movement
                    movement = execute_llm_movement(ser, user_input)
                    if movement:
                        last_movement = movement
            
            except KeyboardInterrupt:
                print("\n[INFO] Interrupted by user")
                break
            except Exception as e:
                log_event("ERROR", f"Error in main loop: {e}")
    
    finally:
        stop_face_recognition()
        if ser and ser != "SIMULATION" and hasattr(ser, 'is_open') and ser.is_open:
            ser.close()
            log_event("INFO", "Serial connection closed")

if __name__ == "__main__":
    main()