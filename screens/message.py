from kivymd.uix.screen import MDScreen
from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.label import MDLabel
import socketio
from datetime import datetime
from screens.db import get_db_cursor
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image

sio = socketio.Client()
connected = False

class Message(MDScreen):
    def on_enter(self):
        self.ids.chat_messages.clear_widgets()

        sender = self.manager.get_screen("Home").ids.home_usernf.title.strip()
        receiver = self.ids.receiver_label.text.strip()

        conn, cur = get_db_cursor()

        # Mark messages sent to current user as seen
        cur.execute("""
            UPDATE chat_data
            SET is_seen = TRUE
            WHERE receiver_user_name = %s AND sender_user_name = %s AND is_seen = FALSE
        """, (sender, receiver))
        conn.commit()

        cur.execute("""
            SELECT sender_user_name, receiver_user_name, message, timestamp, is_seen
            FROM chat_data
            WHERE (sender_user_name = %s AND receiver_user_name = %s) OR (sender_user_name = %s AND receiver_user_name = %s)
            ORDER BY timestamp ASC
        """, (sender, receiver, receiver, sender))

        rows = cur.fetchall()
        for s, r, msg, time, seen in rows:
            self.add_message(s, msg, time, seen)
            

        cur.close()
        conn.close()

    def save_message_to_db(self, sender, receiver, message, timestamp):

        conn, cur = get_db_cursor()

        # Save message
        # cur.execute(
        #     "INSERT INTO chat_data (sender_user_name, receiver_user_name, message, timestamp, is_seen) VALUES (%s, %s, %s, %s, %s)",
        #     (sender, receiver, message, timestamp, False)
        # )

        # Update or insert chatbox (ensure consistent pair order)
        u1, u2 = sorted([sender, receiver])
        cur.execute("""
            INSERT INTO chatbox (sender_user_name, receiver_user_name, last_message, timestamp)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (sender_user_name, receiver_user_name) DO UPDATE
            SET last_message = EXCLUDED.last_message,
                timestamp = EXCLUDED.timestamp;
        """, (u1, u2, message, timestamp))

        conn.commit()
        cur.close()
        conn.close()

    def send_message(self):
        sender = self.manager.get_screen("Home").ids.home_usernf.title.strip()
        receiver = self.ids.receiver_label.text.strip()
        msg = self.ids.chat_input.text.strip()

        print("Sender:", sender)
        print("Receiver:", receiver)
        print("Message:", msg)

        if not sender or not receiver:
            print("Please enter both your username and recipient username.")
            return

        if not msg:
            return

        self.ids.chat_input.text = ""

        # Connect socket if not connected yet (or reconnect for each message)
        if not sio.connected:
            try:
                sio.connect("http://localhost:5000")
                # sio.connect("wss://vc-ser.onrender.com/socket.io")
                # sio.connect("wss://web-production-d416c.up.railway.app")
                sio.emit("join", {"username": sender})
                sio.on("new_private_message", self.receive_message)
            except Exception as e:
                print("Socket connection error:", e)
                return

        timestamp = datetime.now().isoformat()
        self.save_message_to_db(sender, receiver, msg, timestamp)
        self.update_chatbox(receiver, msg, timestamp)


        # Show sent message locally
        self.add_message(sender, msg, timestamp)

        # Emit private message event with dynamic sender and receiver
        sio.emit("private_message", {
            "from": sender,
            "to": receiver,
            "message": msg,
            "time": timestamp
        })

    def receive_message(self, data):
        sender = data.get("from")
        msg = data.get("message")
        time = data.get("time")
        Clock.schedule_once(lambda dt: self.add_message(sender, msg, time))
        # self.save_message_to_db(self, sender, receiver, msg, timestamp)
        self.save_message_to_db(sender, self.manager.get_screen("Home").ids.home_usernf.title.strip(), msg, time)
        self.update_chatbox(sender, msg, time)


    def add_message(self, sender, msg, timestamp, seen=False):
        if isinstance(timestamp, str):
            dt_obj = datetime.fromisoformat(timestamp)
        else:
            dt_obj = timestamp  # already a datetime object from DB

        formatted_time = dt_obj.strftime("%I:%M %p")

        my_username = self.manager.get_screen("Home").ids.home_usernf.title.strip()
        is_self = sender == my_username
        align = "right" if is_self else "left"
        
        message_box = MDBoxLayout(
            orientation="horizontal" if is_self else "horizontal",
            size_hint_y=None,
            height=dp(60),
            padding=dp(5),
            spacing=dp(5),
        )

        label = MDLabel(
            text=f"[b]{'You' if is_self else sender}[/b]: {msg}\n[i]{formatted_time}[/i]",
            markup=True,
            halign=align,
            size_hint_y=None,
            height=dp(60),
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            size_hint_x=0.9
        )

        message_box.add_widget(label)

        if is_self:
            icon_path = "assets/images/seen.png" if seen else "assets/images/send.png"
            seen_icon = Image(
                source=icon_path,
                size_hint=(None, None),
                size=(dp(20), dp(20)),
            )
            message_box.add_widget(seen_icon)

        self.ids.chat_messages.add_widget(message_box)
        Clock.schedule_once(lambda dt: self.ids.scroll_view.scroll_to(message_box), 0.1)

    def update_chatbox(self, other_user, last_msg, timestamp):
        chatbox_screen = self.manager.get_screen("ChatBox")
        chatbox_screen.update_chat_item(other_user, last_msg, timestamp)
