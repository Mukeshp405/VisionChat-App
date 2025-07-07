import cv2
import os
import time
import random
import pygame
import cvzone
import numpy as np
import shutil
from kivymd.uix.screen import MDScreen
from kivy.clock import Clock
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from deepface import DeepFace
import mediapipe as mp
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.app import App
from screens.db import get_db_cursor
from kivymd.uix.filemanager import MDFileManager
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from cvzone.FPS import FPS
from datetime import datetime
from kivy.utils import get_color_from_hex
from kivy.utils import platform
from screens.db import get_db_cursor
from kivy.core.window import Window
from math import sqrt
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from kivymd.uix.button import MDRaisedButton
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from collections import deque, Counter

window_width, window_height = Window.size

# Home
class Home(MDScreen):

    def process(self):

        app = App.get_running_app()

        print("Home username :- ", app.username)
        print("Home password :- ", app.password)

        try:
            conn, cur = get_db_cursor()
            cur.execute("SELECT * FROM user_data WHERE username = %s OR email = %s", (app.username, app.username))
            self.userdetails = cur.fetchone()

            conn.commit()
            cur.close()
            conn.close()

            print("Home userDetails :- ", self.userdetails) 
            img_filename = self.userdetails[5] or "2.png"
            self.userimage = f"assets/uploaded_images/{img_filename}"
            self.username = self.userdetails[2]
            self.firstname = self.userdetails[1]
            self.email = self.userdetails[3]
            self.password = "*" * 10

            nav_drawer = self.manager.get_screen("Home").ids.nav_drawer

            self.manager.get_screen("Home").ids.home_userimage.source = self.userimage
            self.manager.get_screen("Home").ids.home_usernf.title = self.username
            self.manager.get_screen("Home").ids.home_usernf.text = self.firstname
            self.manager.get_screen("Home").ids.top_app_bar.left_action_items = [[self.userimage, lambda x: nav_drawer.set_state("open")]]
            self.manager.get_screen("Account").ids.account_userimage.source = self.userimage
            self.manager.get_screen("Account").ids.account_username.text = self.username
            self.manager.get_screen("Account").ids.account_username2.text = self.username
            self.manager.get_screen("Account").ids.account_firstname.text = self.firstname
            self.manager.get_screen("Account").ids.acc_password.text = self.password
            self.manager.get_screen("Account").ids.acc_email.text = self.email
            self.manager.get_screen("Account").ids.name_name.text = self.firstname
        except Exception as e:
            print(e)

    def open_file_manager(self):
        try:
            self.file_manager = MDFileManager(select_path=self.select_path, exit_manager=self.exit_manager, preview=True)
            self.file_manager.show("/")

            Clock.schedule_once(lambda dt: self.set_toolbar_bg(), 0.2)
        except Exception as e:
            print(e)

    def set_toolbar_bg(self, *args):
        try:
            toolbar = self.file_manager.ids.get("toolbar")
            if toolbar:
                toolbar.md_bg_color = get_color_from_hex("#ad7102") 
                toolbar.specific_text_color = get_color_from_hex("#FFFFFF") 
                print("Toolbar style updated.")
            else:
                print("Toolbar not found in ids.")
        except Exception as e:
            print("Error updating toolbar:", e)
    
    def exit_manager(self, *args):
        self.file_manager.close()
    
    def select_path(self, path):
        try:
            upload_dir = "assets/uploaded_images"
            if not os.path.exists(upload_dir):
                os.mkdir(upload_dir)
            
            filename = os.path.basename(path)
            destination_path = os.path.join(upload_dir, filename)

            shutil.copy(path, destination_path)

            print("File Saved :- ", destination_path)
            print("File filename :- ", filename)
            
            conn, cur = get_db_cursor()
            cur.execute("UPDATE user_data SET img_upload = %s WHERE username = %s", (filename, self.username))

            if self.userimage:
                self.manager.current = "Home"

            print(self.userdetails)
            conn.commit()

            cur.close()
            conn.close()


            self.exit_manager()
            self.userimage = f"assets/uploaded_images/{os.path.basename(path)}"
            self.manager.get_screen("Home").ids.home_userimage.source = self.userimage
            self.manager.get_screen("Home").ids.top_app_bar.left_action_items = [
                [self.userimage, lambda x: self.manager.get_screen("Home").ids.nav_drawer.set_state("open")]
            ]
            self.manager.get_screen("Account").ids.account_userimage.source = self.userimage
        except Exception as e:
            print("File Upload Error :- ", e)

    def handle_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_x = x
            self.click_y = y
    
    count = 0
    apply_facemesh = False
    apply_facedetection = False
    apply_manfilter = False
    apply_dogfilter = False
    apply_birthday1_bg = False
    apply_birthday2_bg = False
    apply_birthday3_bg = False
    apply_birthday4_bg = False
    apply_birthday5_bg = False
    apply_birthday6_bg = False
    apply_background_change = False
    apply_filter1 = False
    apply_filter2 = False
    apply_filter3 = False
    apply_filter4 = False
    apply_filter5 = False
    apply_filter6 = False
    apply_filter7 = False
    apply_filter8 = False
    apply_filter9 = False
    apply_virtual_paint = False
    apply_face_emotion = False
    apply_face_distance = False
    apply_bulloon_pop_game = False
    apply_pong_game = False
    apply_volume_control = False
    click_x = None
    click_y = None

    def on_enter(self):
        try:
            if getattr(self, 'already_loaded', False):
                return
            self.already_loaded = True

            self.process()

            if platform == 'android':
                self.ids.camera_feed.play = True
                Clock.schedule_interval(self.update_android_camera, 1.0 / 30)
            else:
                self.capture = cv2.VideoCapture(0)

                # Alert sound
                pygame.mixer.init()
                self.alert_sound = pygame.mixer.Sound("assets/alert.mp3")
                self.sound_playing = False
                self.hit_sound = pygame.mixer.Sound("assets/short_sound/hit7.wav")
                self.life_lost_sound = pygame.mixer.Sound("assets/short_sound/hit1.wav")
                self.game_over_sound = pygame.mixer.Sound("assets/short_sound/hit11.wav")

                # Detectors and utilities
                self.facedetector = FaceMeshDetector()
                self.handtrack = HandDetector()
                self.segmentor = SelfiSegmentation()
                self.fpsReader = FPS()

                # Face emotion detection
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

                self.facedetector.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1
                )
                self.emotion_history = deque(maxlen=10)  # Stores last 10 emotions
                self.latest_emotion_detect = "Detecting..."

                # Mediapipe Hands Initialization
                self.mpHands = mp.solutions.hands
                self.hands = self.mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
                self.mpDraw = mp.solutions.drawing_utils

                # Audio Control Setup
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                vol_range = self.volume.GetVolumeRange()
                self.minVol = vol_range[0]
                self.maxVol = vol_range[1]
                self.pinky_up = False

                self.restart_btn = Button(
                    text="RESTART",
                    size_hint=(None, None),
                    size=("160dp", "40dp"),
                    pos_hint={"center_x": 0.5, "center_y": 0.4},
                    background_normal='',
                    background_color=get_color_from_hex("#00FF99"),  
                    color=(0, 0, 0, 1),  
                    font_size='20sp',
                    opacity=0,
                    disabled=True,
                )
                self.restart_btn.bind(on_release=self.reset_pong_game_from_button)
                self.add_widget(self.restart_btn)

                # Balloon
                self.detector = HandDetector(detectionCon=0.1, maxHands=1)
                self.apply_bulloon_pop_game = False  # Game is off by default
                self.balloon_img = cv2.imread("assets/images/b1.png", cv2.IMREAD_UNCHANGED)
                self.balloon_img = cv2.resize(self.balloon_img, (80, 110))
                self.balloons = [{'x': random.randint(100, 1100), 'y': random.randint(720, 1000)} for _ in range(3)]
                self.score = 0
                self.speed = 15
                self.balloon_game_over = False
                self.balloon_restart_btn = None
                self.total_time = 60

                # Face Distance Measurement
                self.last_save_time = 0
                self.last_saved_distance = None
                self.lastest_distance_cm = None

                # Face Emotion Detection
                self.latest_emotion_detect = ""

                # Volume Control
                self.latest_volume_level = None

                # Balloon Pop Game
                self.latest_bulloon_score = 0

                Clock.schedule_interval(self.update, 1.0 / 30)

        except Exception as e:
            print("on_enter error:", e)

    # Overlay emoji function (outside Home class)
    def overlay_emoji(self, frame, emoji, x, y):
        if emoji is not None:
            emoji = cv2.resize(emoji, (50, 50))
            h, w, _ = emoji.shape
            try:
                for c in range(0, 3):
                    alpha = emoji[:, :, 3] / 255.0
                    frame[y:y + h, x:x + w, c] = (1. - alpha) * frame[y:y + h, x:x + w, c] + alpha * emoji[:, :, c]
            except:
                pass  

    
    # Helper function to draw labels on the frame
    def draw_label(self, img, text, pos, bg_color):
        font_scale = 0.8
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        x, y = pos
        cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), bg_color, -1)
        cv2.putText(img, text, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    def reset_pong_game_from_button(self, *args):
        self.ball = True
        self.lives = 3
        self.myscore = 0
        self.level = 1
        self.deltax = 10
        self.deltay = -10
        self.xpos = self.frame.shape[1] // 2
        self.ypos = self.frame.shape[0] // 2
        self.game_over_sound_played = False

        # Hide the button again
        self.restart_btn.opacity = 0
        self.restart_btn.disabled = True
    
    # pong game data store in database
    def save_or_update_pong_highscore(self, user_id, new_score):
        conn, cur = get_db_cursor()
        # Step 1: Check existing high score
        cur.execute("SELECT high_score FROM pong_gesture_game WHERE userID = %s ORDER BY timestamp DESC LIMIT 1", (user_id,))
        result = cur.fetchone()

        if result:
            existing_highscore = result[0]
            if new_score > existing_highscore:
                # Step 2: Update high score
                cur.execute("""
                    UPDATE pong_gesture_game 
                    SET high_score = %s, timestamp = NOW()
                    WHERE userID = %s AND high_score = %s
                """, (new_score, user_id, existing_highscore))
                print(f"High score updated from {existing_highscore} to {new_score}")
            else:
                print(f"New score {new_score} is not higher than existing high score {existing_highscore}, no update.")
        else:
            # No record yet, insert new high score
            cur.execute("""
                INSERT INTO pong_gesture_game (userID, high_score, timestamp) VALUES (%s, %s, NOW())
            """, (user_id, new_score))
            print(f"High score saved for the first time: {new_score}")

        conn.commit()
        cur.close()
        conn.close()

    # face emotion data store in database
    def save_or_update_face_emotion(self, user_id, face_emotion):
        conn, cur = get_db_cursor()
        cur.execute("SELECT id FROM face_emotion_detection WHERE userID = %s", (user_id,))
        result = cur.fetchone()

        if result:
            cur.execute("""
                        UPDATE face_emotion_detection
                        SET emotionType = %s
                        WHERE userID = %s
                        """, (face_emotion, user_id))
        else:
            cur.execute("""
                        INSERT INTO face_emotion_detection (userID, emotionType)
                        VALUES (%s, %s)
                        """, (user_id, face_emotion))
        conn.commit()
        cur.close()
        conn.close()

    # face distance data store in database
    def save_or_update_face_distance(self, user_id, distance_cm):
        conn, cur = get_db_cursor()

        # Check if record already exists for this user
        cur.execute("""
            SELECT id FROM face_distance_measurement WHERE userID = %s
        """, (user_id,))
        result = cur.fetchone()

        if result:
            # If record exists, update the value
            cur.execute("""
                UPDATE face_distance_measurement
                SET distance_value = %s
                WHERE userID = %s
            """, (distance_cm, user_id))
        else:
            # Else insert a new one
            cur.execute("""
                INSERT INTO face_distance_measurement (userID, distance_value)
                VALUES (%s, %s)
            """, (user_id, distance_cm))

        conn.commit()
        cur.close()
        conn.close()

    # Volume control data store in database
    def save_or_update_volume_control(self, user_id, volume_level):
        conn, cur = get_db_cursor()

        # Check if record already exists for this user
        cur.execute("""
            SELECT id FROM volume_control WHERE userID = %s
        """, (user_id,))
        result = cur.fetchone()

        if result:
            # If record exists, update the value
            cur.execute("""
                UPDATE volume_control
                SET volume_level = %s
                WHERE userID = %s
            """, (volume_level, user_id))
        else:
            # Else insert a new one
            cur.execute("""
                INSERT INTO volume_control (userID, volume_level)
                VALUES (%s, %s)
            """, (user_id, volume_level))

        conn.commit()
        cur.close()
        conn.close()
    
    # Virtual paint data store in database
    def save_or_update_paint_color(self, user_id, paint_color_tuple):
        paint_color = ",".join(map(str, paint_color_tuple))  # Convert (R, G, B) → "255,0,255"

        conn, cur = get_db_cursor()

        cur.execute("SELECT id FROM virtual_paint WHERE userID = %s", (user_id,))
        result = cur.fetchone()

        if result:
            cur.execute("""
                UPDATE virtual_paint
                SET paintColor = %s
                WHERE userID = %s
            """, (paint_color, user_id))
        else:
            cur.execute("""
                INSERT INTO virtual_paint (userID, paintColor)
                VALUES (%s, %s)
            """, (user_id, paint_color))

        conn.commit()
        cur.close()
        conn.close()

    # Balloon pop game data store in database
    def save_balloon_pop_high_score_to_db(self, user_id, new_score):
        try:
            conn, cursor = get_db_cursor()
            # Check if a record exists
            cursor.execute("SELECT high_score FROM balloon_gesture_game WHERE userID = %s", (user_id,))
            result = cursor.fetchone()

            if result:
                old_score = result[0]
                if new_score > old_score:
                    cursor.execute("""
                        UPDATE balloon_gesture_game
                        SET high_score = %s, timestamp = CURRENT_TIMESTAMP
                        WHERE userID = %s
                    """, (new_score, user_id))
            else:
                cursor.execute("""
                    INSERT INTO balloon_gesture_game (userID, high_score)
                    VALUES (%s, %s)
                """, (user_id, new_score))

            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print("Error saving high score:", e)
    
    def get_user_id_from_db(self, username):
        conn, cur = get_db_cursor()
        # cursor = conn.cursor()
        query = "SELECT id FROM user_data WHERE username = %s"
        cur.execute(query, (username,))
        result = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        if result:
            return result[0]  # user_id
        return None

    # Balloon
    def start_balloon_game(self):
        self.start_time = time.time() 
        self.start_time = time.time()
        self.score = 0
        self.speed = 15
        self.balloon_game_over = False
        self.balloons = [{'x': random.randint(100, 1100), 'y': random.randint(720, 1000)} for _ in range(3)]
        self.initialize_balloons(count=10)
 
    def restart_game(self, instance):
        self.apply_bulloon_pop_game = True
        self.start_balloon_game()  # ✅ Restart timer and game values

        if self.restart_btn:
            self.ids.main_layout.remove_widget(self.restart_btn)
            self.restart_btn = None

    def overlay_image_alpha(self, img, overlay_img):
        """Overlay an image (with alpha channel) on the img."""
        if overlay_img.shape[2] == 4:
            alpha_mask = overlay_img[:, :, 3] / 255.0
            for c in range(0, 3):
                img[:, :, c] = img[:, :, c] * (1 - alpha_mask) + overlay_img[:, :, c] * alpha_mask
        return img

    def show_restart_button(self):
        if not self.restart_btn:
            self.restart_btn = MDRaisedButton(
                text='Restart',
                pos_hint={'center_x': 0.5, 'center_y': 0.2},
                size_hint=(0.3, 0.1),
                on_release=self.restart_game
            )
            self.ids.main_layout.add_widget(self.restart_btn)

    def reset_balloon(self, rect):
        rect['x'] = random.randint(100, 1100)
        rect['y'] = random.randint(720, 1000)

    def initialize_balloons(self, count=10):  # Increase count from default (e.g., 5 to 10)
        self.balloons = []
        frame_width = self.frame.shape[1]
        for _ in range(count):
            x = random.randint(50, frame_width - self.balloon_img.shape[1] - 50)
            y = random.randint(self.frame.shape[0] // 2, self.frame.shape[0])
            self.balloons.append({'x': x, 'y': y})
    def update(self, dt):
        try:
            ret, self.frame = self.capture.read()
            self.frame = cv2.flip(self.frame, 1)

            if not ret or self.frame is None:
                print("Error: Failed to read from camera.")
                return
            if ret:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                results = None

                # ==================================

                # Face Emotion Detection
                if self.apply_face_emotion:
                    rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                    # FaceMesh detection
                    results2 = self.facedetector.face_mesh.process(rgb_frame)

                    emotion = "Detecting..."

                    if results2.multi_face_landmarks:
                        for face_landmarks in results2.multi_face_landmarks:
                            ih, iw, _ = self.frame.shape
                            x_list, y_list = [], []

                            for lm in face_landmarks.landmark:
                                x, y = int(lm.x * iw), int(lm.y * ih)
                                x_list.append(x)
                                y_list.append(y)

                            x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, iw)
                            y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, ih)

                            face_roi = self.frame[y_min:y_max, x_min:x_max]

                            if face_roi.size > 0:
                                try:
                                    face_resized = cv2.resize(face_roi, (224, 224))
                                    result = DeepFace.analyze(
                                        face_resized,
                                        actions=['emotion'],
                                        enforce_detection=False,
                                        detector_backend='mediapipe'
                                    )
                                    emotion = result[0]['dominant_emotion']
                                    confidence = result[0]['emotion'][emotion]

                                    # Optional: confidence threshold
                                    if confidence > 0.5:
                                        self.emotion_history.append(emotion)
                                except Exception as e:
                                    print("Emotion detection error:", e)
                                    self.emotion_history.append("Unknown")

                            # Use the most common emotion from recent frames
                            if self.emotion_history:
                                common_emotion = Counter(self.emotion_history).most_common(1)[0][0]
                                self.latest_emotion_detect = common_emotion
                            else:
                                self.latest_emotion_detect = "Detecting..."

                            # Draw bounding box and label
                            cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 3, cv2.LINE_AA)
                            self.draw_label(self.frame, self.latest_emotion_detect, (x_min, y_min), (255, 0, 255))

                # Pong Game
                if self.apply_pong_game:
                    frame_height, frame_width = self.frame.shape[:2]

                    if not hasattr(self, 'pong_initialized'):
                        self.ball = True
                        self.lives = 3
                        self.myscore = 0
                        self.highscore = 0
                        self.level = 1
                        self.deltax = 10
                        self.deltay = -10
                        self.xpos = frame_width // 2
                        self.ypos = frame_height // 2
                        self.prevval = frame_width // 2
                        self.paddlewidth = 100
                        self.paddleheight = 15
                        self.paddlecolor = (0, 255, 204)
                        self.bgcolor = (0, 102, 255)
                        self.mpHands = mp.solutions.hands
                        self.hands = self.mpHands.Hands(max_num_hands=1)
                        self.game_over_sound_played = False
                        self.pong_initialized = True
                        self.click_x = None
                        self.click_y = None
                        self.highscore_saved = False

                    # Apply translucent orange overlay
                    overlay = self.frame.copy()
                    orange_color = (0, 140, 255)
                    cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), orange_color, -1)
                    alpha = 0.3
                    self.frame = cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0)

                    results = self.hands.process(cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
                    val = 0
                    if results.multi_hand_landmarks:
                        lm = results.multi_hand_landmarks[0].landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
                        val = int(lm.x * frame_width)
                    else:
                        val = self.prevval

                    self.prevval = val
                    val = np.clip(val, self.paddlewidth // 2, frame_width - self.paddlewidth // 2)
                    paddle_x = val

                    # Draw paddle
                    cv2.rectangle(self.frame,(paddle_x - self.paddlewidth // 2, frame_height - self.paddleheight),(paddle_x + self.paddlewidth // 2, frame_height),self.paddlecolor, -1)

                    if self.ball:
                        cv2.circle(self.frame, (self.xpos, self.ypos), 6, (255, 255, 255), -1)
                        self.xpos += self.deltax
                        self.ypos += self.deltay

                    if self.xpos >= frame_width - 5 or self.xpos <= 5:
                        self.deltax = -self.deltax
                    if self.ypos <= 5:
                        self.deltay = -self.deltay

                    # Ball hits paddle
                    if paddle_x - self.paddlewidth // 2 <= self.xpos <= paddle_x + self.paddlewidth // 2:
                        if frame_height - self.paddleheight - 5 <= self.ypos <= frame_height:
                            self.deltay = -self.deltay
                            self.myscore += 1
                            pygame.mixer.Sound.play(self.hit_sound)
                            if self.myscore % 5 == 0:
                                self.level += 1
                                self.deltax += 1 if self.deltax > 0 else -1
                                self.deltay += 1 if self.deltay > 0 else -1

                    # Ball falls
                    if self.ypos >= frame_height:
                        self.lives -= 1
                        pygame.mixer.Sound.play(self.life_lost_sound)
                        self.xpos = frame_width // 2
                        self.ypos = frame_height // 2
                        if self.deltay > 0:
                            self.deltay = -self.deltay

                    # Game Over
                    if self.lives == 0:
                        self.ball = False
                        if not self.game_over_sound_played:
                            pygame.mixer.Sound.play(self.game_over_sound)
                            self.game_over_sound_played = True

                        cv2.putText(self.frame, "GAME OVER", (frame_width // 2 - 130, frame_height // 2 - 40),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                        if self.myscore > self.highscore:
                            self.highscore = self.myscore

                        # Scores
                        cv2.putText(self.frame, f"Score: {self.myscore}", (frame_width // 2 - 100, frame_height // 2),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(self.frame, f"High Score: {self.highscore}", (frame_width // 2 - 100, frame_height // 2 + 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        if self.lives == 0 and not self.highscore_saved:
                            self.user_id = self.get_user_id_from_db(self.username)
                            self.save_or_update_pong_highscore(self.user_id, self.highscore)
                            self.highscore_saved = True

                        self.restart_btn.opacity = 1
                        self.restart_btn.disabled = False

                    cv2.putText(self.frame, f"Lives: {self.lives}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(self.frame, f"Score: {self.myscore}", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(self.frame, f"Level: {self.level}", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # === Balloon Pop Game ===
                if self.apply_bulloon_pop_game:
                    time_remain = int(self.total_time - (time.time() - self.start_time))

                    # print("Time left:", time_remain)

                    if time_remain >= 0:
                        self.balloon_game_over = False

                        hands, self.frame = self.detector.findHands(self.frame, flipType=False)

                        # Dim background slightly like during Game Over (but lighter)
                        dim_overlay = self.frame.copy()
                        dim_color = (0, 0, 0)
                        alpha = 0.2  # Adjust this value (0.2 - 0.4 is subtle dimming, similar to light opacity)
                        cv2.rectangle(dim_overlay, (0, 0), (self.frame.shape[1], self.frame.shape[0]), dim_color, -1)
                        cv2.addWeighted(dim_overlay, alpha, self.frame, 1 - alpha, 0, self.frame)

                        # Move and draw balloons
                        for rect in self.balloons:
                            rect['y'] -= self.speed
                            if rect['y'] < -100:
                                self.reset_balloon(rect)

                            # Check for hand collision
                            if hands:
                                x, y = hands[0]['lmList'][8][:2]
                                if rect['x'] < x < rect['x'] + self.balloon_img.shape[1] and \
                                rect['y'] < y < rect['y'] + self.balloon_img.shape[0]:
                                    self.reset_balloon(rect)
                                    self.score += 10
                                    self.speed += 1

                            # Overlay balloon image on frame
                            h, w, _ = self.balloon_img.shape
                            bh, bw = rect['y'], rect['x']
                            try:
                                overlay = self.balloon_img
                                if 0 <= bh < self.frame.shape[0] - h and 0 <= bw < self.frame.shape[1] - w:
                                    self.frame[bh:bh + h, bw:bw + w] = self.overlay_image_alpha(
                                        self.frame[bh:bh + h, bw:bw + w], overlay
                                    )
                            except:
                                pass

                        # Draw score and time
                        cv2.putText(self.frame, f'Score: {self.score}', (40, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(self.frame, f'Time: {time_remain}s', (40, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                    else:
                        if not self.balloon_game_over:
                            self.balloon_game_over = True
                            print("Game Over")
                            self.current_user_id = self.get_user_id_from_db(self.username)
                            # self.save_balloon_pop_high_score_to_db(user_id=self.current_user_id, new_score=self.score)
                            # self.latest_bulloon_score = self.score
                            self.show_restart_button()


                        # Dim background
                        overlay = self.frame.copy()
                        dim_color = (0, 0, 0)
                        alpha = 0.5  # Lower value = darker background
                        cv2.rectangle(overlay, (0, 0), (self.frame.shape[1], self.frame.shape[0]), dim_color, -1)
                        cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0, self.frame)

                        # Display "Time Up" and final score
                        cv2.putText(self.frame, 'Time UP!', (190, 250),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        cv2.putText(self.frame, f'Final Score: {self.score}', (190, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
                        self.latest_bulloon_score = self.score
                        
                        
                # Face Distance Measurement
                if self.apply_face_distance:
                    self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)
                    if face_mesh:
                        face_points = face_mesh[0]
                        left_eye = face_points[145]
                        right_eye = face_points[374]

                        pixel_distance, _ = self.facedetector.findDistance(left_eye, right_eye)

                        real_width = 6.3  
                        focal = 530 
                        distance_cm = (real_width * focal) / pixel_distance

                        if distance_cm < 50:
                            if not self.sound_playing:
                                self.alert_sound.play(-1)
                                self.sound_playing = True
                            color = (255, 0, 0)
                        else:
                            if self.sound_playing:
                                self.alert_sound.stop()
                                self.sound_playing = False
                            color = (0, 255, 0)

                        center_x = (left_eye[0] + right_eye[0]) // 2
                        top_y = min(left_eye[1], right_eye[1]) - 60

                        cvzone.putTextRect(self.frame, f'Distance: {int(distance_cm)} cm',
                                        (center_x - 100, top_y), scale=2,
                                        colorR=color, font=cv2.FONT_HERSHEY_PLAIN)
                        
                        # current_time = time.time()
                        self.lastest_distance_cm = distance_cm

                        # if (self.last_saved_distance is None or abs(distance_cm - self.last_saved_distance) > 2) and (current_time - self.last_save_time > 5):
                        #     self.save_or_update_face_distance(self.user_id, distance_cm)  # Your function to insert/update in DB
                        #     self.last_saved_distance = distance_cm
                        #     self.last_save_time = current_time


                # Virtual Paint
                if self.apply_virtual_paint:
                    try:
                        # --- Initialization (Only once) ---
                        if not hasattr(self, 'vp_initialized'):
                            print("Virtual Paint Initialized")
                            self.vp_initialized = True
                            self.brushThickness = 15
                            self.eraserThickness = 100
                            self.header_index = 0
                            folderPath = "Images"
                            myList = os.listdir(folderPath)
                            self.overlayList = [cv2.imread(f'{folderPath}/{imPath}') for imPath in myList]
                            self.header = self.overlayList[0]
                            self.drawColor = (255, 0, 255)
                            self.xp, self.yp = 0, 0
                            self.imgCanvas = np.zeros_like(self.frame, np.uint8)
                            self.hand_detector = HandDetector(detectionCon=0.85, maxHands=1)

                        # --- Per Frame Drawing ---
                        hands, self.frame = self.hand_detector.findHands(self.frame, draw=False)
                        if hands:
                            lmList = hands[0]['lmList']
                            x1, y1 = lmList[8][:2]
                            x2, y2 = lmList[12][:2]
                            fingers = self.hand_detector.fingersUp(hands[0])

                            if fingers[1] and fingers[2]:  # Selection Mode
                                self.xp, self.yp = 0, 0
                                if y1 < 125:
                                    frame_width = self.frame.shape[1]
                                    button_width = frame_width // 4
                                    if 0 < x1 < button_width:
                                        self.header = self.overlayList[0]
                                        self.drawColor = (255, 0, 255)
                                    elif button_width < x1 < button_width * 2:
                                        self.header = self.overlayList[1]
                                        self.drawColor = (255, 0, 0)
                                    elif button_width * 2 < x1 < button_width * 3:
                                        self.header = self.overlayList[2]
                                        self.drawColor = (0, 255, 0)
                                    elif button_width * 3 < x1 < frame_width:
                                        self.header = self.overlayList[3]
                                        self.drawColor = (0, 0, 0)
                                cv2.rectangle(self.frame, (x1, y1 - 25), (x2, y2 + 25), self.drawColor, cv2.FILLED)

                            elif fingers[1] and not fingers[2]:  # Drawing Mode
                                cv2.circle(self.frame, (x1, y1), 15, self.drawColor, cv2.FILLED)
                                if self.xp == 0 and self.yp == 0:
                                    self.xp, self.yp = x1, y1
                                thickness = self.eraserThickness if self.drawColor == (0, 0, 0) else self.brushThickness
                                cv2.line(self.frame, (self.xp, self.yp), (x1, y1), self.drawColor, thickness)
                                cv2.line(self.imgCanvas, (self.xp, self.yp), (x1, y1), self.drawColor, thickness)
                                self.xp, self.yp = x1, y1

                        # Merge canvas with current frame
                        imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
                        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
                        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
                        self.frame = cv2.bitwise_and(self.frame, imgInv)
                        self.frame = cv2.bitwise_or(self.frame, self.imgCanvas)

                        # Resize and overlay header
                        resized_header = cv2.resize(self.header, (self.frame.shape[1], 125))
                        self.frame[0:125, 0:self.frame.shape[1]] = resized_header

                    except Exception as e:
                        print("Virtual Paint Error:", e)

                # Gesture Volume Control
                if self.apply_volume_control:
                    results = self.hands.process(self.frame)

                    if results and results.multi_hand_landmarks:
                        handLms = results.multi_hand_landmarks[0]
                        h, w, _ = self.frame.shape
                        lmList = []

                        for id, lm in enumerate(handLms.landmark):
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append((id, cx, cy))

                        if len(lmList) >= 21:
                            x1, y1 = lmList[4][1], lmList[4][2]
                            x2, y2 = lmList[8][1], lmList[8][2]
                            length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                            pinky_tip_y = lmList[20][2]
                            pinky_base_y = lmList[17][2]
                            self.pinky_up = pinky_tip_y < pinky_base_y

                            vol = np.interp(length, [30, 180], [self.minVol, self.maxVol])
                            vol_bar = np.interp(length, [30, 180], [400, 150])
                            vol_per = np.interp(length, [30, 180], [0, 100])

                            if self.pinky_up:
                                self.volume.SetMasterVolumeLevel(vol, None)

                            cv2.circle(self.frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
                            cv2.circle(self.frame, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
                            cv2.line(self.frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.circle(self.frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

                            getVol = int(self.volume.GetMasterVolumeLevelScalar() * 100)
                            bar_height = np.interp(getVol, [0, 100], [400, 150])
                            cv2.rectangle(self.frame, (50, 150), (85, 400), (255, 0, 0), 2)
                            cv2.rectangle(self.frame, (50, int(bar_height)), (85, 400), (255, 0, 0), cv2.FILLED)
                            cv2.putText(self.frame, f"{getVol}%", (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # 🔹 Volume percentage at top-left corner
                            cv2.putText(self.frame, f"Volume: {getVol}%", (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

                            self.latest_volume_level = getVol

                # Always draw landmarks if detected
                if results and results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        self.mpDraw.draw_landmarks(self.frame, handLms, self.mpHands.HAND_CONNECTIONS)

                # ==================================

                # cat filter
                if self.apply_facemesh:
                    try:
                        if not hasattr(self, "filter2_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/cat.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter10_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 1.3)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 1.36)

                            resized_overlay = cv2.resize(self.filter10_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.43)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 2 overlay:", e)

                # FaceDetection
                if self.apply_facedetection:
                    try:
                        # Bilateral filter for smooth shading
                        smooth = cv2.bilateralFilter(self.frame, d=9, sigmaColor=75, sigmaSpace=75)

                        # Edge detection
                        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
                        _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
                        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                        # Combine smooth and edges
                        toon = cv2.bitwise_and(smooth, mask_colored)

                        # Color enhancement
                        hsv = cv2.cvtColor(toon, cv2.COLOR_BGR2HSV)
                        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 25)  # increase saturation
                        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 15)  # increase brightness
                        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                        self.frame = result
                    except Exception as e:
                        print(e)
                
                # Man Filter
                if self.apply_manfilter:
                    try:
                        img1 = cv2.imread(r"assets\images\p8.png")
                    except FileNotFoundError as fe:
                        print("Image not found :- ", fe)

                    try:
                        if img1 is None:
                            print("Error: img1 not loaded.")
                            return

                        # Detect face landmarks
                        img1, faces1 = self.facedetector.findFaceMesh(img1, draw=False)

                        img2 = self.frame
                        if img2 is None:
                            print("Error: img2 is None.")
                            return

                        img2, faces2 = self.facedetector.findFaceMesh(img2, draw=False)

                        if faces1 and faces2:
                            landmarks_points1 = np.array(faces1[0], np.int32)
                            landmarks_points2 = np.array(faces2[0], np.int32)

                            # Convex hulls for both faces
                            convexhull1 = cv2.convexHull(landmarks_points1)
                            convexhull2 = cv2.convexHull(landmarks_points2)

                            # Delaunay triangulation
                            rect = cv2.boundingRect(convexhull1)
                            subdiv = cv2.Subdiv2D(rect)
                            subdiv.insert(landmarks_points1.tolist())
                            triangles = subdiv.getTriangleList()
                            triangles = np.array(triangles, dtype=np.int32)

                            indexes_triangles = []
                            for t in triangles:
                                pt1, pt2, pt3 = (t[:2], t[2:4], t[4:])
                                indexes_triangles.append([
                                    np.argmin(np.linalg.norm(landmarks_points1 - pt1, axis=1)),
                                    np.argmin(np.linalg.norm(landmarks_points1 - pt2, axis=1)),
                                    np.argmin(np.linalg.norm(landmarks_points1 - pt3, axis=1))
                                ])

                            # Face swap process
                            img2_new_face = np.zeros_like(img2, np.uint8)

                            for triangle_index in indexes_triangles:
                                tr1_pts = landmarks_points1[triangle_index]
                                tr2_pts = landmarks_points2[triangle_index]

                                matrix = cv2.getAffineTransform(np.float32(tr1_pts), np.float32(tr2_pts))
                                warped_triangle = cv2.warpAffine(img1, matrix, (img2.shape[1], img2.shape[0]))

                                mask_triangle = np.zeros_like(img2, np.uint8)
                                cv2.fillConvexPoly(mask_triangle, np.int32(tr2_pts), (255, 255, 255))

                                img2_new_face = cv2.bitwise_and(img2_new_face, cv2.bitwise_not(mask_triangle))
                                img2_new_face = cv2.add(img2_new_face, cv2.bitwise_and(warped_triangle, mask_triangle))

                            # Create mask for seamless cloning
                            mask = np.zeros(img2.shape[:2], dtype=np.uint8)
                            mask = cv2.fillConvexPoly(mask, convexhull2, 255)

                            # Compute center for seamless cloning
                            center = tuple(np.mean(convexhull2, axis=0, dtype=int)[0])
                            result = cv2.seamlessClone(img2_new_face, img2, mask, center, cv2.MIXED_CLONE)

                            self.frame = result
                    except Exception as e:
                        print(e)

                # Man2 Filter
                if self.apply_dogfilter:
                    try:
                        img1 = cv2.imread(r"assets\images\p7.png")
                    except FileNotFoundError as fe:
                        print("Image not found :- ", fe)

                    try:
                        if img1 is None:
                            print("Error: img1 not loaded.")
                            return

                        # Detect face landmarks
                        img1, faces1 = self.facedetector.findFaceMesh(img1, draw=False)

                        img2 = self.frame
                        if img2 is None:
                            print("Error: img2 is None.")
                            return

                        img2, faces2 = self.facedetector.findFaceMesh(img2, draw=False)

                        if faces1 and faces2:
                            landmarks_points1 = np.array(faces1[0], np.int32)
                            landmarks_points2 = np.array(faces2[0], np.int32)

                            # Convex hulls for both faces
                            convexhull1 = cv2.convexHull(landmarks_points1)
                            convexhull2 = cv2.convexHull(landmarks_points2)

                            # Delaunay triangulation
                            rect = cv2.boundingRect(convexhull1)
                            subdiv = cv2.Subdiv2D(rect)
                            subdiv.insert(landmarks_points1.tolist())
                            triangles = subdiv.getTriangleList()
                            triangles = np.array(triangles, dtype=np.int32)

                            indexes_triangles = []
                            for t in triangles:
                                pt1, pt2, pt3 = (t[:2], t[2:4], t[4:])
                                indexes_triangles.append([
                                    np.argmin(np.linalg.norm(landmarks_points1 - pt1, axis=1)),
                                    np.argmin(np.linalg.norm(landmarks_points1 - pt2, axis=1)),
                                    np.argmin(np.linalg.norm(landmarks_points1 - pt3, axis=1))
                                ])

                            # Face swap process
                            img2_new_face = np.zeros_like(img2, np.uint8)

                            for triangle_index in indexes_triangles:
                                tr1_pts = landmarks_points1[triangle_index]
                                tr2_pts = landmarks_points2[triangle_index]

                                matrix = cv2.getAffineTransform(np.float32(tr1_pts), np.float32(tr2_pts))
                                warped_triangle = cv2.warpAffine(img1, matrix, (img2.shape[1], img2.shape[0]))

                                mask_triangle = np.zeros_like(img2, np.uint8)
                                cv2.fillConvexPoly(mask_triangle, np.int32(tr2_pts), (255, 255, 255))

                                img2_new_face = cv2.bitwise_and(img2_new_face, cv2.bitwise_not(mask_triangle))
                                img2_new_face = cv2.add(img2_new_face, cv2.bitwise_and(warped_triangle, mask_triangle))

                            # Create mask for seamless cloning
                            mask = np.zeros(img2.shape[:2], dtype=np.uint8)
                            mask = cv2.fillConvexPoly(mask, convexhull2, 255)

                            # Compute center for seamless cloning
                            center = tuple(np.mean(convexhull2, axis=0, dtype=int)[0])
                            result = cv2.seamlessClone(img2_new_face, img2, mask, center, cv2.MIXED_CLONE)

                            self.frame = result
                    except Exception as e:
                        print(e)

                # Background Change
                if self.apply_background_change:
                    try:
                        # Load background image
                        if not hasattr(self, "background_change"):
                            bg_path = "assets/FiltersImages/bg3.png" 
                            self.background_bg = cv2.imread(bg_path)
                            self.background_bg = cv2.resize(self.background_bg, (self.frame.shape[1], self.frame.shape[0]))

                        imgOut = self.segmentor.removeBG(self.frame, self.background_bg)
                        self.frame = imgOut
                    except Exception as e:
                        print("Error applying Happy Birthday background:", e)
                
                # Happy Birthday background
                if self.apply_birthday1_bg:
                    try:
                        # Load overlay
                        if not hasattr(self, "birthday1_fg"):
                            overlay_path = "assets/FiltersImages/hb1.png" 
                            self.birthday_overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                            self.birthday_overlay = cv2.resize(self.birthday_overlay, (self.frame.shape[1], self.frame.shape[0]))

                        # Split overlay into color + alpha
                        if self.birthday_overlay.shape[2] == 4:
                            b, g, r, a = cv2.split(self.birthday_overlay)
                            overlay_color = cv2.merge((r, g, b))
                            mask = cv2.merge((a, a, a)).astype(float) / 255.0

                            background = self.frame.astype(float)
                            foreground = overlay_color.astype(float)

                            # Blend transparent overlay over the processed image
                            final = cv2.multiply(mask, foreground) + cv2.multiply(1.0 - mask, background)
                            self.frame = final.astype(np.uint8)
                        else:
                            print("Warning: PNG overlay has no alpha channel. Skipping overlay.")
                            self.frame = self.frame

                    except Exception as e:
                        print("Error applying birthday overlay:", e)

                # Happy Birthday 2 background
                if self.apply_birthday2_bg:
                    try:
                        # Load overlay
                        if not hasattr(self, "birthday2_fg"):
                            overlay_path = "assets/FiltersImages/hb3.png" 
                            self.birthday_overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                            self.birthday_overlay = cv2.resize(self.birthday_overlay, (self.frame.shape[1], self.frame.shape[0]))

                        # Split overlay into color + alpha
                        if self.birthday_overlay.shape[2] == 4:
                            b, g, r, a = cv2.split(self.birthday_overlay)
                            overlay_color = cv2.merge((r, g, b))
                            mask = cv2.merge((a, a, a)).astype(float) / 255.0

                            background = self.frame.astype(float)
                            foreground = overlay_color.astype(float)

                            # Blend transparent overlay over the processed image
                            final = cv2.multiply(mask, foreground) + cv2.multiply(1.0 - mask, background)
                            self.frame = final.astype(np.uint8)
                        else:
                            print("Warning: PNG overlay has no alpha channel. Skipping overlay.")
                            self.frame = self.frame

                    except Exception as e:
                        print("Error applying birthday overlay:", e)

                # Happy Birthday 3 background
                if self.apply_birthday3_bg:
                    try:
                        # Load overlay
                        if not hasattr(self, "birthday3_fg"):
                            overlay_path = "assets/FiltersImages/hb5.png" 
                            self.birthday_overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                            self.birthday_overlay = cv2.resize(self.birthday_overlay, (self.frame.shape[1], self.frame.shape[0]))

                        # Split overlay into color + alpha
                        if self.birthday_overlay.shape[2] == 4:
                            b, g, r, a = cv2.split(self.birthday_overlay)
                            overlay_color = cv2.merge((r, g, b))
                            mask = cv2.merge((a, a, a)).astype(float) / 255.0

                            background = self.frame.astype(float)
                            foreground = overlay_color.astype(float)

                            # Blend transparent overlay over the processed image
                            final = cv2.multiply(mask, foreground) + cv2.multiply(1.0 - mask, background)
                            self.frame = final.astype(np.uint8)
                        else:
                            print("Warning: PNG overlay has no alpha channel. Skipping overlay.")
                            self.frame = self.frame

                    except Exception as e:
                        print("Error applying birthday overlay:", e)

                # Happy Birthday 4 background
                if self.apply_birthday4_bg:
                    try:
                        # Load overlay 
                        if not hasattr(self, "birthday4_fg"):
                            overlay_path = "assets/FiltersImages/hb7.png" 
                            self.birthday_overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                            self.birthday_overlay = cv2.resize(self.birthday_overlay, (self.frame.shape[1], self.frame.shape[0]))

                        # Split overlay into color + alpha
                        if self.birthday_overlay.shape[2] == 4:
                            b, g, r, a = cv2.split(self.birthday_overlay)
                            overlay_color = cv2.merge((r, g, b))
                            mask = cv2.merge((a, a, a)).astype(float) / 255.0

                            background = self.frame.astype(float)
                            foreground = overlay_color.astype(float)

                            # Blend transparent overlay over the processed image
                            final = cv2.multiply(mask, foreground) + cv2.multiply(1.0 - mask, background)
                            self.frame = final.astype(np.uint8)
                        else:
                            print("Warning: PNG overlay has no alpha channel. Skipping overlay.")
                            self.frame = self.frame

                    except Exception as e:
                        print("Error applying birthday overlay:", e)

                # Happy Birthday 5 background
                if self.apply_birthday5_bg:
                    try:
                        # Load overlay 
                        if not hasattr(self, "birthday5_fg"):
                            overlay_path = "assets/FiltersImages/hb9.png" 
                            self.birthday_overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                            self.birthday_overlay = cv2.resize(self.birthday_overlay, (self.frame.shape[1], self.frame.shape[0]))

                        # Split overlay into color + alpha
                        if self.birthday_overlay.shape[2] == 4:
                            b, g, r, a = cv2.split(self.birthday_overlay)
                            overlay_color = cv2.merge((r, g, b))
                            mask = cv2.merge((a, a, a)).astype(float) / 255.0

                            background = self.frame.astype(float)
                            foreground = overlay_color.astype(float)

                            # Blend transparent overlay over the processed image
                            final = cv2.multiply(mask, foreground) + cv2.multiply(1.0 - mask, background)
                            self.frame = final.astype(np.uint8)
                        else:
                            print("Warning: PNG overlay has no alpha channel. Skipping overlay.")
                            self.frame = self.frame

                    except Exception as e:
                        print("Error applying birthday overlay:", e)

                # Happy Birthday 6 background
                if self.apply_birthday6_bg:
                    try:
                        # Load overlay
                        if not hasattr(self, "birthday6_fg"):
                            overlay_path = "assets/FiltersImages/hb11.png" 
                            self.birthday_overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
                            self.birthday_overlay = cv2.resize(self.birthday_overlay, (self.frame.shape[1], self.frame.shape[0]))

                        # Split overlay into color + alpha
                        if self.birthday_overlay.shape[2] == 4:
                            b, g, r, a = cv2.split(self.birthday_overlay)
                            overlay_color = cv2.merge((r, g, b))
                            mask = cv2.merge((a, a, a)).astype(float) / 255.0

                            background = self.frame.astype(float)
                            foreground = overlay_color.astype(float)

                            # Blend transparent overlay over the processed image
                            final = cv2.multiply(mask, foreground) + cv2.multiply(1.0 - mask, background)
                            self.frame = final.astype(np.uint8)
                        else:
                            print("Warning: PNG overlay has no alpha channel. Skipping overlay.")
                            self.frame = self.frame

                    except Exception as e:
                        print("Error applying birthday overlay:", e)
                
                # Filter 1
                if self.apply_filter1:
                    try:
                        if not hasattr(self, "filter1_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/6.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter1_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 1.3)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 0.5)

                            resized_overlay = cv2.resize(self.filter1_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 1.0)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 1 overlay:", e)
               
               # Filter 2
                if self.apply_filter2:
                    try:
                        if not hasattr(self, "filter2_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/1.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter2_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 1.4)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 1.9)

                            resized_overlay = cv2.resize(self.filter2_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.43)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 2 overlay:", e)
               
               # Filter 3
                if self.apply_filter3:
                    try:
                        if not hasattr(self, "filter3_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/8.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter3_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 1.4)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 0.5)

                            resized_overlay = cv2.resize(self.filter3_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 1)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 3 overlay:", e)
               
               # Filter 4
                if self.apply_filter4:
                    try:
                        if not hasattr(self, "filter4_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/3.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter4_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 2.8)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 2.0)

                            resized_overlay = cv2.resize(self.filter4_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.32)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 4 overlay:", e)
               
               # Filter 5
                if self.apply_filter5:
                    try:
                        if not hasattr(self, "filter5_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/13.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter5_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 1.4)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 1.1)

                            resized_overlay = cv2.resize(self.filter5_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.04)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 5 overlay:", e)
               
               # Filter 6
                if self.apply_filter6:
                    try:
                        if not hasattr(self, "filter6_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/sun.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter6_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 1.6)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 1.3)

                            resized_overlay = cv2.resize(self.filter6_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.28)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 6 overlay:", e)
               
               # Filter 7
                if self.apply_filter7:
                    try:
                        if not hasattr(self, "filter7_overlay"):
                            # Load pirate background
                            bg_img = cv2.imread("assets/FiltersImages/Images/bg3.png")
                            if bg_img is None:
                                print("Could not load background image.")
                                return

                            self.pirate_bg = cv2.resize(bg_img, (self.frame.shape[1], self.frame.shape[0]))

                            # Load pirate hat overlay
                            overlay_img = cv2.imread("assets/FiltersImages/Images/pirate.png", cv2.IMREAD_UNCHANGED)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter7_overlay = overlay_img
                            self.segmentor = SelfiSegmentation()  # Initialize segmentor

                        # Remove background and replace with pirate island
                        self.pirate_bg = cv2.resize(self.pirate_bg, (self.frame.shape[1], self.frame.shape[0]))
                        self.frame = self.segmentor.removeBG(self.frame, self.pirate_bg, cutThreshold=0.6)

                        # Detect face mesh
                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 2.8)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 1.9)

                            resized_overlay = cv2.resize(self.filter7_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.4)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 7 overlay:", e)
               
               # Filter 8
                if self.apply_filter8:
                    try:
                        if not hasattr(self, "filter8_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/beard.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter8_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 2.3)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 2.0)

                            resized_overlay = cv2.resize(self.filter8_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.38)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 8 overlay:", e)
               
               # Filter 9
                if self.apply_filter9:
                    try:
                        if not hasattr(self, "filter9_overlay"):
                            overlay_img = cv2.imread("assets/FiltersImages/Images/2.png", cv2.IMREAD_UNCHANGED)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2BGRA)
                            if overlay_img is None:
                                print("Could not load overlay image. Check the file path.")
                                return

                            if overlay_img.shape[2] != 4:
                                print("Overlay image doesn't have an alpha channel. Expected BGRA format.")
                                return

                            self.filter9_overlay = overlay_img

                        self.frame, face_mesh = self.facedetector.findFaceMesh(self.frame, draw=False)

                        if face_mesh:
                            face = face_mesh[0]

                            forehead = face[10]
                            chin = face[152]
                            left = face[234]
                            right = face[454]

                            head_width = int(np.linalg.norm(np.array(left) - np.array(right)) * 2.5)
                            head_height = int(np.linalg.norm(np.array(forehead) - np.array(chin)) * 1.5)

                            resized_overlay = cv2.resize(self.filter9_overlay, (head_width, head_height), interpolation=cv2.INTER_AREA)

                            x = forehead[0] - head_width // 2
                            y = forehead[1] - int(head_height * 0.64)

                            self.frame = cvzone.overlayPNG(self.frame, resized_overlay, [x, y])

                    except Exception as e:
                        print("Error applying filter 9 overlay:", e)


                # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.frame = cv2.flip(self.frame, 0)

                buf = self.frame.tobytes()
                texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

                self.ids.camera_feed.texture = texture
        except Exception as e:
            print(e)

    # =========================

    # Cancle All Feature
    def cancel_all_feature(self):
        self.apply_facemesh = False
        self.apply_facedetection = False
        self.apply_manfilter = False
        self.apply_dogfilter = False
        self.apply_birthday1_bg = False
        self.apply_birthday2_bg = False
        self.apply_birthday3_bg = False
        self.apply_birthday4_bg = False
        self.apply_birthday5_bg = False
        self.apply_birthday6_bg = False
        self.apply_background_change = False
        self.apply_filter1 = False
        self.apply_filter2 = False
        self.apply_filter3 = False
        self.apply_filter4 = False
        self.apply_filter5 = False
        self.apply_filter6 = False
        self.apply_filter7 = False
        self.apply_filter8 = False
        self.apply_filter9 = False
        self.apply_virtual_paint = False
        if not self.apply_virtual_paint:
            if hasattr(self, 'drawColor') and isinstance(self.drawColor, tuple):
                self.user_id = self.get_user_id_from_db(self.username)
                self.save_or_update_paint_color(self.user_id, self.drawColor)
                self.drawColor = None
        self.apply_face_emotion = False
        if not self.apply_face_emotion:
            if not self.apply_face_emotion and self.latest_emotion_detect is not None:
                self.user_id = self.get_user_id_from_db(self.username)
                self.save_or_update_face_emotion(self.user_id, self.latest_emotion_detect)
                self.latest_emotion_detect = None
        self.apply_face_distance = False
        if not self.apply_face_distance:
            if not self.apply_face_distance and self.lastest_distance_cm is not None:
                self.user_id = self.get_user_id_from_db(self.username)
                self.save_or_update_face_distance(self.user_id, self.lastest_distance_cm)
                self.lastest_distance_cm = None
        self.apply_bulloon_pop_game = False
        if not self.apply_bulloon_pop_game and self.latest_bulloon_score is not None:
            self.user_id = self.get_user_id_from_db(self.username)
            self.save_balloon_pop_high_score_to_db(self.user_id, self.latest_bulloon_score)
            self.latest_bulloon_score = None

        self.apply_pong_game = False
        self.apply_volume_control = False
        if not self.apply_volume_control:
            if not self.apply_volume_control and self.latest_volume_level is not None:
                self.user_id = self.get_user_id_from_db(self.username)
                self.save_or_update_volume_control(self.user_id, self.latest_volume_level)
                self.latest_volume_level = None

    # Face Emotion
    def toggle_face_emotion(self):
        self.apply_face_emotion = True
        self.manager.current = "Home"
        self.manager.transition.direction = "down"
        if self.apply_pong_game:
            self.apply_pong_game = not self.apply_pong_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_bulloon_pop_game:
            self.apply_bulloon_pop_game = not self.apply_bulloon_pop_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_face_distance:
            self.apply_face_distance = not self.apply_face_distance
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_virtual_paint:
            self.apply_virtual_paint = not self.apply_virtual_paint
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_volume_control:
            self.apply_volume_control = not self.apply_volume_control
            self.manager.current = "Home"
            self.manager.transition.direction = "down"

    # Pong Game
    def toggle_pong_game(self):
        self.apply_pong_game = True
        self.manager.current = "Home"
        self.manager.transition.direction = "down"
        if self.apply_face_emotion:
            self.apply_face_emotion = not self.apply_face_emotion
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_bulloon_pop_game:
            self.apply_bulloon_pop_game = not self.apply_bulloon_pop_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_face_distance:
            self.apply_face_distance = not self.apply_face_distance
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_virtual_paint:
            self.apply_virtual_paint = not self.apply_virtual_paint
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_volume_control:
            self.apply_volume_control = not self.apply_volume_control
            self.manager.current = "Home"
            self.manager.transition.direction = "down"

    # Bulloon Pop Game
    def toggle_bulloon_pop_game(self):
        if not self.apply_bulloon_pop_game:
            self.apply_bulloon_pop_game = True
            self.start_balloon_game()
        self.manager.current = "Home"
        self.manager.transition.direction = "down"
        if self.apply_face_emotion:
            self.apply_face_emotion = not self.apply_face_emotion
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_pong_game:
            self.apply_pong_game = not self.apply_pong_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_face_distance:
            self.apply_face_distance = not self.apply_face_distance
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_virtual_paint:
            self.apply_virtual_paint = not self.apply_virtual_paint
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_volume_control:
            self.apply_volume_control = not self.apply_volume_control
            self.manager.current = "Home"
            self.manager.transition.direction = "down"

    # Face Distance Measurement
    def toggle_face_distance(self):
        self.apply_face_distance = True
        self.manager.current = "Home"
        self.manager.transition.direction = "down"
        if self.apply_face_emotion:
            self.apply_face_emotion = not self.apply_face_emotion
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_bulloon_pop_game:
            self.apply_bulloon_pop_game = not self.apply_bulloon_pop_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_pong_game:
            self.apply_pong_game = not self.apply_pong_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_virtual_paint:
            self.apply_virtual_paint = not self.apply_virtual_paint
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_volume_control:
            self.apply_volume_control = not self.apply_volume_control
            self.manager.current = "Home"
            self.manager.transition.direction = "down"

    # VirtualPaint
    def toggle_virtual_paint(self):
        self.apply_virtual_paint = True
        self.manager.current = "Home"
        self.manager.transition.direction = "down"
        if self.apply_face_emotion:
            self.apply_face_emotion = not self.apply_face_emotion
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_bulloon_pop_game:
            self.apply_bulloon_pop_game = not self.apply_bulloon_pop_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_face_distance:
            self.apply_face_distance = not self.apply_face_distance
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_pong_game:
            self.apply_pong_game = not self.apply_pong_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_volume_control:
            self.apply_volume_control = not self.apply_volume_control
            self.manager.current = "Home"
            self.manager.transition.direction = "down"

    # Gesture Volume Control
    def toggle_volume_control(self):
        self.apply_volume_control = True
        self.manager.current = "Home"
        self.manager.transition.direction = "down"
        if self.apply_face_emotion:
            self.apply_face_emotion = not self.apply_face_emotion
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_bulloon_pop_game:
            self.apply_bulloon_pop_game = not self.apply_bulloon_pop_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_face_distance:
            self.apply_face_distance = not self.apply_face_distance
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_pong_game:
            self.apply_pong_game = not self.apply_pong_game
            self.manager.current = "Home"
            self.manager.transition.direction = "down"
        elif self.apply_virtual_paint:
            self.apply_virtual_paint = not self.apply_virtual_paint
            self.manager.current = "Home"
            self.manager.transition.direction = "down"

    # ===========================

    # FaceMesh
    def toggle_facemesh(self):
        try:
            self.apply_facemesh = not self.apply_facemesh
            if self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Cartoon
    def toggle_facedetection(self):
        try:
            self.apply_facedetection = not self.apply_facedetection
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Man Filter
    def toggle_manfilter(self):
        try:
            self.apply_manfilter = not self.apply_manfilter
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
        except Exception as e:
            print(e)

    # Man2 Filter
    def toggle_dogfilter(self):
        try:
            self.apply_dogfilter = not self.apply_dogfilter
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Happy Birthday 1 Foreground
    def toggle_birthday1_fg(self):
        try:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Happy Birthday 2 Foreground
    def toggle_birthday2_fg(self):
        try:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Happy Birthday 3 Foreground
    def toggle_birthday3_fg(self):
        try:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Happy Birthday 4 Foreground
    def toggle_birthday4_fg(self):
        try:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Happy Birthday 5 Foreground
    def toggle_birthday5_fg(self):
        try:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Happy Birthday 6 Foreground
    def toggle_birthday6_fg(self):
        try:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_background_change:
                self.apply_background_change = not self.apply_background_change
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Background Change
    def toggle_background_change(self):
        try:
            self.apply_background_change = not self.apply_background_change
            if self.apply_facemesh:
                self.apply_facemesh = not self.apply_facemesh
            elif self.apply_facedetection:
                self.apply_facedetection = not self.apply_facedetection
            elif self.apply_manfilter:
                self.apply_manfilter = not self.apply_manfilter
            elif self.apply_dogfilter:
                self.apply_dogfilter = not self.apply_dogfilter
            elif self.apply_birthday1_bg:
                self.apply_birthday1_bg = not self.apply_birthday1_bg
            elif self.apply_birthday2_bg:
                self.apply_birthday2_bg = not self.apply_birthday2_bg
            elif self.apply_birthday3_bg:
                self.apply_birthday3_bg = not self.apply_birthday3_bg
            elif self.apply_birthday4_bg:
                self.apply_birthday4_bg = not self.apply_birthday4_bg
            elif self.apply_birthday5_bg:
                self.apply_birthday5_bg = not self.apply_birthday5_bg
            elif self.apply_birthday6_bg:
                self.apply_birthday6_bg = not self.apply_birthday6_bg
            elif self.apply_filter1:
                self.apply_filter1 = not self.apply_filter1
            elif self.apply_filter2:
                self.apply_filter2 = not self.apply_filter2
            elif self.apply_filter3:
                self.apply_filter3 = not self.apply_filter3
            elif self.apply_filter4:
                self.apply_filter4 = not self.apply_filter4
            elif self.apply_filter5:
                self.apply_filter5 = not self.apply_filter5
            elif self.apply_filter6:
                self.apply_filter6 = not self.apply_filter6
            elif self.apply_filter7:
                self.apply_filter7 = not self.apply_filter7
            elif self.apply_filter8:
                self.apply_filter8 = not self.apply_filter8
            elif self.apply_filter9:
                self.apply_filter9 = not self.apply_filter9
        except Exception as e:
            print(e)

    # Filter 1
    def toggle_filter1(self):
        self.apply_filter1 = not self.apply_filter1
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 2
    def toggle_filter2(self):
        self.apply_filter2 = not self.apply_filter2
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 3
    def toggle_filter3(self):
        self.apply_filter3 = not self.apply_filter3
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 4
    def toggle_filter4(self):
        self.apply_filter4 = not self.apply_filter4
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 5
    def toggle_filter5(self):
        self.apply_filter5 = not self.apply_filter5
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 6
    def toggle_filter6(self):
        self.apply_filter6 = not self.apply_filter6
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 7
    def toggle_filter7(self):
        self.apply_filter7 = not self.apply_filter7
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 8
    def toggle_filter8(self):
        self.apply_filter8 = not self.apply_filter8
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1
        elif self.apply_filter9:
            self.apply_filter9 = not self.apply_filter9

    # Filter 9
    def toggle_filter9(self):
        self.apply_filter9 = not self.apply_filter9
        if self.apply_facemesh:
            self.apply_facemesh = not self.apply_facemesh
        elif self.apply_facedetection:
            self.apply_facedetection = not self.apply_facedetection
        elif self.apply_manfilter:
            self.apply_manfilter = not self.apply_manfilter
        elif self.apply_dogfilter:
            self.apply_dogfilter = not self.apply_dogfilter
        elif self.apply_background_change:
            self.apply_background_change = not self.apply_background_change
        elif self.apply_birthday1_bg:
            self.apply_birthday1_bg = not self.apply_birthday1_bg
        elif self.apply_birthday2_bg:
            self.apply_birthday2_bg = not self.apply_birthday2_bg
        elif self.apply_birthday3_bg:
            self.apply_birthday3_bg = not self.apply_birthday3_bg
        elif self.apply_birthday4_bg:
            self.apply_birthday4_bg = not self.apply_birthday4_bg
        elif self.apply_birthday5_bg:
            self.apply_birthday5_bg = not self.apply_birthday5_bg
        elif self.apply_birthday6_bg:
            self.apply_birthday6_bg = not self.apply_birthday6_bg
        elif self.apply_filter2:
            self.apply_filter2 = not self.apply_filter2
        elif self.apply_filter3:
            self.apply_filter3 = not self.apply_filter3
        elif self.apply_filter4:
            self.apply_filter4 = not self.apply_filter4
        elif self.apply_filter5:
            self.apply_filter5 = not self.apply_filter5
        elif self.apply_filter6:
            self.apply_filter6 = not self.apply_filter6
        elif self.apply_filter7:
            self.apply_filter7 = not self.apply_filter7
        elif self.apply_filter8:
            self.apply_filter8 = not self.apply_filter8
        elif self.apply_filter1:
            self.apply_filter1 = not self.apply_filter1

    # Capture Images
    def capture_image(self):
        try:
            if hasattr(self, 'frame'):
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                self.frame = cv2.flip(self.frame, 0)

                save_dir = r"assets/captured_images"
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"IMG{timestamp}.png"
                filepath = os.path.join(save_dir, filename)

                cv2.imwrite(filepath, self.frame)
                # print(f"Image captured: {filename}")

                # Save image info to MySQL database
                try:
                    app = App.get_running_app()
                    conn, cur = get_db_cursor()

                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS captured_images (
                            id SERIAL PRIMARY KEY,
                            username VARCHAR(255),
                            image_name VARCHAR(255) UNIQUE,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    insert_query = """
                        INSERT INTO captured_images (username, image_name)
                        VALUES (%s, %s)
                    """
                    cur.execute(insert_query, (app.username, filename))

                    conn.commit()
                    cur.close()
                    conn.close()

                    print(f"Image record inserted to DB for user: {app.username}")
                except Exception as e:
                    print("Database error while saving captured image:", e)
        except Exception as e:
            print(e)
