from kivy.properties import StringProperty
from kivymd.uix.screen import MDScreen

class UserDetail(MDScreen):
    username = StringProperty()
    def set_user_data(self, username, image_path, fullname):
        self.ids.username_label.text = username
        self.ids.username_label2.text = username
        self.ids.user_image.source = image_path
        self.ids.fullname_label.text = fullname
        self.ids.fullname_label2.text = fullname
    
    def send_to_message(self):
        self.manager.current = "Message"
        message_screen = self.manager.get_screen("Message")
        message_screen.ids.sender_input.text = self.username
        message_screen.ids.receiver_input.text = self.username