from kivymd.uix.screen import MDScreen
from kivymd.uix.list import OneLineAvatarListItem, ImageLeftWidget
from kivy.clock import Clock
from screens.db import get_db_cursor
import os

class Search(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.users = []  # cache users as list of dicts

    def on_pre_enter(self, *args):
        Clock.schedule_once(self.load_users, 0)

    def load_users(self, dt):
        conn, cursor = get_db_cursor()
        cursor.execute('SELECT username, firstname, img_upload FROM user_data')
        self.users = [{'username': row[0], 'firstname': row[1], 'img_upload': row[2]} for row in cursor.fetchall()]
        conn.close()

        self.search_users("1")  # Show all users initially

    def search_users(self, text):
        user_list = self.ids.user_list
        user_list.clear_widgets()

        for user_data in self.users:
            username = user_data['username']
            firstname = user_data['firstname']
            img_upload = user_data['img_upload']

            if text.lower() in username.lower():
                image_path = (
                    f"assets/uploaded_images/{img_upload}"
                    if img_upload and os.path.exists(f"assets/uploaded_images/{img_upload}")
                    else "assets/uploaded_images/2.png"
                )

                fullname = firstname

                item = OneLineAvatarListItem(
                    text=username,
                    theme_text_color="Custom",
                    text_color="white",
                    on_release=lambda x, u=username, img=image_path, fn=fullname: self.open_user_detail(u, img, fn)
                )
                item.add_widget(ImageLeftWidget(source=image_path))
                user_list.add_widget(item)

    def open_user_detail(self, username, image_path, fullname):
        detail_screen = self.manager.get_screen("UserDetail")
        detail_screen.set_user_data(username, image_path, fullname)
        self.manager.current = "UserDetail"
