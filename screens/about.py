from kivymd.uix.screen import MDScreen
import webbrowser

class About(MDScreen):
    def open_url(self, url):
        webbrowser.open(url)