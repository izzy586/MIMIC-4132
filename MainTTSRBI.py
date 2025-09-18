

import openai
import time
import json
import serial
import os
import pyttsx3
import random
from datetime import datetime

# ------------------------
# CONFIG VARIABLES 
# ------------------------
openai.api_key = "YOUR_OPENAI_API_KEY"

# DO NOT EDIT BELOW THIS LINE

SERIAL_PORT = "COM3"
BAUD_RATE = 115200

SERVO_MIN = 0
SERVO_MAX = 180
STEP_DELAY = 0.05
INTERPOLATION_STEPS = 20
DEFAULT_TTS_RATE = 150
DEFAULT_TTS_VOLUME = 1.0

# DO NOT EDIT ABOVE THIS LINE

# Servo pins DO NOT EDIT PRECONFIG
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

# Predefined Movements DO NOT EDIT PRECONFIG
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
    }
}

SEQUENCES_DIR = "sequences"
os.makedirs(SEQUENCES_DIR, exist_ok=True)

# ------------------------
# INITIALIZE TTS
# ------------------------
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", DEFAULT_TTS_RATE)
tts_engine.setProperty("volume", DEFAULT_TTS_VOLUME)

# ------------------------
# SERIAL FUNCTIONS
# ------------------------
def connect_serial(port, baud):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print(f"[INFO] Connected to {port} at {baud} baud")
        return ser
    except Exception as e:
        print(f"[ERROR] Serial connection failed: {e}")
        return None

def send_servo_command(ser, servo_name, position):
    position = max(SERVO_MIN, min(SERVO_MAX, int(position)))
    command = f"{servo_name}:{position}\n"
    ser.write(command.encode())
    time.sleep(STEP_DELAY)
    print(f"[DEBUG] Servo {servo_name} -> {position}")

# ------------------------
# MOVEMENT UTILITIES
# ------------------------
def interpolate_positions(start, end, steps=INTERPOLATION_STEPS):
    for i in range(steps):
        yield int(start + (end - start) * (i / (steps-1)))

def execute_movement(ser, movement):
    max_steps = max(len(positions) for positions in movement.values() if isinstance(positions, list))
    previous_positions = {servo: positions[0] for servo, positions in movement.items() if isinstance(positions, list)}

    for step in range(max_steps):
        for servo, positions in movement.items():
            if isinstance(positions, list):
                target = positions[step % len(positions)]
                for pos in interpolate_positions(previous_positions.get(servo, target), target):
                    send_servo_command(ser, servo, pos)
                previous_positions[servo] = target
        time.sleep(0.1)

# ------------------------
# TTS FUNCTION
# ------------------------
def speak_text(text):
    print(f"[TTS] {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# ------------------------
# LLM INTEGRATION
# ------------------------
def ask_llm(prompt):
    system_message = """
You are an assistant that generates safe servo movement sequences for a 3D-printed animatronic.
Return a JSON object with servo names as keys and lists of positions (0-180 degrees) as values.
Include a 'speech' key for TTS output.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=700
    )
    answer_text = response.choices[0].message['content'].strip()
    try:
        return json.loads(answer_text)
    except json.JSONDecodeError:
        print(f"[ERROR] LLM returned invalid JSON: {answer_text}")
        return None

def execute_llm_movement(ser, prompt):
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
    filename = os.path.join(SEQUENCES_DIR, f"{name}.json")
    with open(filename, "w") as f:
        json.dump(movement, f, indent=4)
    print(f"[INFO] Sequence '{name}' saved")

def load_sequence(name):
    filename = os.path.join(SEQUENCES_DIR, f"{name}.json")
    if not os.path.exists(filename):
        print(f"[ERROR] Sequence '{name}' not found")
        return None
    with open(filename, "r") as f:
        movement = json.load(f)
    return movement

# ------------------------
# DEBUG LOGGING
# ------------------------
def log_event(event_type, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{event_type}] {message}")

# ------------------------
# RANDOM MOVEMENTS (EXTRA VARS)
# ------------------------
def random_servo_movement():
    movement = {}
    for servo in SERVOS.keys():
        movement[servo] = [random.randint(SERVO_MIN, SERVO_MAX) for _ in range(random.randint(2,5))]
    movement["speech"] = "Random movement activated"
    return movement

# ------------------------
# MAIN INTERACTIVE LOOP
# ------------------------
def main():
    ser = connect_serial(SERIAL_PORT, BAUD_RATE)
    if not ser:
        return
    last_movement = None

    print("[INFO] Welcome to the Mimic Animatronic Controller with TTS and LLM!")
    print("Commands: predefined moves, AI description, save <name>, load <name>, random, exit")

    while True:
        user_input = input("Command: ").strip()
        if user_input.lower() == "exit":
            log_event("INFO", "Exiting program")
            break
        elif user_input.startswith("save "):
            if last_movement:
                save_sequence(user_input[5:], last_movement)
            else:
                log_event("WARNING", "No movement to save")
        elif user_input.startswith("load "):
            movement = load_sequence(user_input[5:])
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
            movement = MOVEMENTS[user_input]
            execute_movement(ser, movement)
            last_movement = movement
        else:
            movement = execute_llm_movement(ser, user_input)
            if movement:
                last_movement = movement

if __name__ == "__main__":
    main()