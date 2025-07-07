from kivy.config import Config
Config.set('graphics', 'resizable', False)
# Config.set('graphics', 'borderless', '1')
from kivy.properties import StringProperty
from kivy.uix.filechooser import ScreenManager
from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.clock import Clock
from screens.dashboard import DashBoard
from screens.home import Home
from screens.login import Login
from screens.register import Register
from screens.start_page import Start_page
from screens.account import Account
from screens.setting import Setting
from screens.help import Help
from screens.about import About
from screens.features import Features
from screens.otp_page import OTP
from screens.search import Search
from screens.user_details import UserDetail
from screens.message import Message 
from screens.chatbox import ChatBox
from screens.capturedImagesGallery import CapturedImagesGallery

Window.size = (740, 710)
Window.set_icon("assets/images/favicon2.png")

# VisionChat
class VisionChat(MDApp):
    username = StringProperty(allownone=True)
    # password = StringProperty(allownone=True)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user = Home()
    
    def build(self, *args):
        try:
            screen_manager = ScreenManager()

            # Load separate kv files
            Builder.load_file("kivy/about.kv")
            Builder.load_file("kivy/account.kv")
            Builder.load_file("kivy/dashboard.kv")  
            Builder.load_file("kivy/help.kv")
            Builder.load_file("kivy/features.kv")
            Builder.load_file("kivy/home.kv") 
            Builder.load_file("kivy/login.kv")
            Builder.load_file("kivy/register.kv")
            Builder.load_file("kivy/setting.kv")
            Builder.load_file("kivy/start_page.kv")
            Builder.load_file("kivy/otp_page.kv")
            Builder.load_file("kivy/search.kv")
            Builder.load_file("kivy/userdetails.kv")
            Builder.load_file("kivy/message.kv")
            Builder.load_file("kivy/chatbox.kv")
            Builder.load_file("kivy/capturedImagesGallery.kv")

            screen_manager.add_widget(Start_page(name="Start_page"))
            screen_manager.add_widget(Login(name="Login"))
            screen_manager.add_widget(Register(name="Register"))
            screen_manager.add_widget(DashBoard(name="Dashboard"))
            screen_manager.add_widget(Home(name="Home"))
            screen_manager.add_widget(Setting(name="Setting"))
            screen_manager.add_widget(Account(name="Account"))
            screen_manager.add_widget(Help(name="Help"))
            screen_manager.add_widget(About(name="About"))
            screen_manager.add_widget(Features(name="Features"))
            screen_manager.add_widget(OTP(name="OTP_Page"))
            screen_manager.add_widget(Search(name="Search"))
            screen_manager.add_widget(UserDetail(name="UserDetail"))
            screen_manager.add_widget(Message(name="Message"))
            screen_manager.add_widget(ChatBox(name="ChatBox"))
            screen_manager.add_widget(CapturedImagesGallery(name="Gallery"))

            return screen_manager
        except Exception as e:
            print(e)

    def clear_user_session(self):
        self.username = ""
        self.password = ""

    def set_logged_in_user(self, username):
        self.username = username
    
    def on_start(self):
        try:
            Clock.schedule_once(self.start_page, 7)
        except Exception as e:
            print(e)
    
    def start_page(self, *args):
        try:
            self.root.current = "Dashboard"
        except Exception as e:
            print(e)

if __name__ == "__main__":   
    VisionChat().run()