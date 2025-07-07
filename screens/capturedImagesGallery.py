import os
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.fitimage import FitImage
from kivymd.uix.label import MDLabel
from kivy.app import App
from screens.db import get_db_cursor
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.modalview import ModalView

class CapturedImagesGallery(MDScreen):
    def on_pre_enter(self):
        grid = self.ids.gallery_grid
        screen_width = Window.width
        item_width = 160
        grid.cols = max(2, int(screen_width // item_width))

    def on_enter(self):
        try:
            self.load_images()
        except Exception as e:
            print(e)

    def load_images(self):
        app = App.get_running_app()
        try:
            conn, cur = get_db_cursor()
            cur.execute("SELECT image_name FROM captured_images WHERE username = %s ORDER BY timestamp DESC", (app.username,))
            images = cur.fetchall()
            cur.close()
            conn.close()

            grid = self.ids.gallery_grid
            grid.clear_widgets()

            for (image_name,) in images:
                full_path = os.path.join(r"assets\captured_images", image_name)
                if os.path.exists(full_path):
                    card = MDCard(orientation="vertical", size_hint_y=None, height="240dp", radius=[20])
                    img = FitImage(source=full_path, size_hint=(1, .85))
                    label = MDLabel(text=image_name, halign="center", size_hint=(1, .15))

                    # Attach touch/click handler
                    img.bind(on_touch_down=lambda instance, touch, path=full_path: self.show_fullscreen_image(path) if instance.collide_point(*touch.pos) else None)

                    card.add_widget(img)
                    card.add_widget(label)
                    grid.add_widget(card)
                else:
                    print(f"Image not found on disk: {full_path}")

        except Exception as e:
            print("Error loading images from database:", e)

    def show_fullscreen_image(self, image_path):
        # Create a modal popup to display the image
        modal = ModalView(size_hint=(0.9, 0.9), auto_dismiss=True)
        img = Image(source=image_path, allow_stretch=True, keep_ratio=True)
        modal.add_widget(img)
        modal.open()