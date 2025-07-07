from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivy.metrics import dp
from datetime import datetime
import os
from screens.db import get_db_cursor
from kivy.uix.image import Image

class ChatBox(MDScreen):
    chat_items = {}

    def on_enter(self):
        self.ids.chat_list.clear_widgets()
        self.chat_items.clear()
        current_user = self.manager.get_screen("Home").ids.home_usernf.title.strip()
        self.load_chatbox_data(current_user)

    def load_chatbox_data(self, current_user):
        conn, cur = get_db_cursor()

        cur.execute("""
            SELECT
                CASE WHEN sender_user_name = %s THEN receiver_user_name ELSE sender_user_name END AS other_user,
                last_message,
                timestamp
            FROM chatbox
            WHERE sender_user_name = %s OR receiver_user_name = %s
            ORDER BY timestamp DESC
        """, (current_user, current_user, current_user))

        rows = cur.fetchall()
        for user, msg, time in rows:
            self.update_chat_item(user, msg, time.isoformat())

        cur.close()
        conn.close()

    def update_chat_item(self, other_user, last_msg, timestamp):
        conn, cur = get_db_cursor()

        cur.execute("SELECT img_upload FROM user_data WHERE username = %s", (other_user,))
        result = cur.fetchone()

        if result and result[0]:
            image_filename = result[0]
        else:
            image_filename = "2.png"  # fallback image

        formatted_time = datetime.fromisoformat(timestamp).strftime("%I:%M %p")
        display_text = f"[b]{other_user}[/b]\n{last_msg}\n[i]{formatted_time}[/i]"

        if other_user in self.chat_items:
            # Update existing item
            self.chat_items[other_user].children[0].text = display_text
        else:
            image_path = f"assets/uploaded_images/{image_filename}"
            if not os.path.exists(image_path):
                image_path = "assets/uploaded_images/2.png"

            layout = MDBoxLayout(
                orientation="horizontal",
                size_hint_y=None,
                height=dp(80),
                padding=(dp(10), dp(10)),
                spacing=dp(10),
            )

            profile_pic = Image(
                source=image_path,
                size_hint=(None, None),
                size=(dp(50), dp(50)),
                allow_stretch=True,
                keep_ratio=True,
            )

            label = MDLabel(
                text=display_text,
                markup=True,
                theme_text_color="Custom",
                text_color=(1, 1, 1, 1),
                halign="left",
            )

            layout.add_widget(profile_pic)
            layout.add_widget(label)
            layout.bind(on_touch_down=self.on_chat_select)

            self.ids.chat_list.add_widget(layout)
            self.chat_items[other_user] = layout

        cur.close()
        conn.close()

    def on_chat_select(self, instance, touch):
        if instance.collide_point(*touch.pos):
            for child in instance.children:
                if isinstance(child, MDLabel):
                    selected_user = child.text.split("\n")[0].replace("[b]", "").replace("[/b]", "")
                    message_screen = self.manager.get_screen("Message")
                    message_screen.ids.receiver_label.text = selected_user
                    self.manager.current = "Message"
                    break

