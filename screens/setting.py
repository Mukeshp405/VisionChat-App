from kivymd.uix.banner.banner import MDFlatButton
from kivymd.uix.screen import MDScreen
from kivymd.uix.dialog import MDDialog
from kivy.app import App
from screens.db import get_db_cursor


class Setting(MDScreen):
    # Change theme color
    def change_theme(self):
        if self.manager.get_screen("Setting").ids.theme_button.icon == "weather-sunny":
            # Setting
            self.manager.get_screen("Setting").ids.SettingContainer.md_bg_color = "#999999"
            self.manager.get_screen("Setting").ids.theme_button.icon = "weather-night"

            # self.screenSetting = self.manager.get_screen("Setting")
            # self.text_id_set = ["theme_name", "email_outline_icon", "support_email"]


            # for widget_id in self.text_id_set:
            #     self.screenSetting.ids[widget_id].text_color = "black"
            
            # Account
            self.manager.get_screen("Account").ids.AccountContainer.md_bg_color = "#999999"
            # self.screenAccount = self.manager.get_screen("Account")
            # self.text_id_acc = ["account_firstname", "password_name", "lock_outline_icon", "username_name", "mobile_name", "cellphone_basic_icon", "email_name", "pass_lock_outline_icon", "email_outline_icon"]

            # for widget_id_acc in self.text_id_acc:
            #     self.screenAccount.ids[widget_id_acc].text_color = "black"
            
            # Home
            self.manager.get_screen("Home").ids.nav_drawer.md_bg_color = "#999999"
            # self.manager.get_screen("Home").ids.home_usernf.text_color = "black"
            # self.manager.get_screen("Home").ids.account_circle_outline_home.text_color = "black"
            # self.manager.get_screen("Home").ids.account_circle_outline_home.icon_color = "black"
            # self.manager.get_screen("Home").ids.cog_outline_home.text_color = "black"
            # self.manager.get_screen("Home").ids.cog_outline_home.icon_color = "black"
            # self.manager.get_screen("Home").ids.help_circle_outline_home.text_color = "black"
            # self.manager.get_screen("Home").ids.help_circle_outline_home.icon_color = "black"
            # self.manager.get_screen("Home").ids.information_outline_home.text_color = "black"
            # self.manager.get_screen("Home").ids.information_outline_home.icon_color = "black"


        else:
            # Setting
            self.manager.get_screen("Setting").ids.SettingContainer.md_bg_color = "#1c1c1c"
            self.manager.get_screen("Setting").ids.theme_button.icon = "weather-sunny"

            # for widget_id in self.text_id_set:
            #     self.screenSetting.ids[widget_id].text_color = "white"

            # Account
            self.manager.get_screen("Account").ids.AccountContainer.md_bg_color = "#1c1c1c"

            # for widget_id_acc in self.text_id_acc:
            #     self.screenAccount.ids[widget_id_acc].text_color = "white"

            # Home
            self.manager.get_screen("Home").ids.nav_drawer.md_bg_color = "#1c1c1c"
            # self.manager.get_screen("Home").ids.home_usernf.text_color = "white"
            # self.manager.get_screen("Home").ids.account_circle_outline_home.text_color = "white"
            # self.manager.get_screen("Home").ids.account_circle_outline_home.icon_color = "white"
            # self.manager.get_screen("Home").ids.cog_outline_home.text_color = "white"
            # self.manager.get_screen("Home").ids.cog_outline_home.icon_color = "white"
            # self.manager.get_screen("Home").ids.help_circle_outline_home.text_color = "white"
            # self.manager.get_screen("Home").ids.help_circle_outline_home.icon_color = "white"
            # self.manager.get_screen("Home").ids.information_outline_home.text_color = "white"
            # self.manager.get_screen("Home").ids.information_outline_home.icon_color = "white"
    
    # Delete Account
    def delete_account(self):
        ok_btn = MDFlatButton(text="Yes", on_release=self.ok_delete_account)
        cancel_btn = MDFlatButton(text="Cancel", on_release=self.cancel_delete_account)
        self.deleted_acc = MDDialog(title="Delete Account", text="Are your sure, you want to delete your account.", size_hint=(0.7, 0.2), buttons=[cancel_btn, ok_btn])
        self.deleted_acc.open()

    # Ok Delete Account
    def ok_delete_account(self, obj):
        try:
            app = App.get_running_app()
            conn, cur = get_db_cursor()

            print("Delete username :- ", app.username)
            print("Delete password :- ", app.password)


            cur.execute("DELETE FROM user_data WHERE username = %s", (app.username,))
            cur.execute("DELETE FROM captured_images WHERE username = %s", (app.username,))
            print("Deleted")
            conn.commit()

            cur.close()
            conn.close()

            # self.ok_logout_account(obj)

            app.username = ""
            app.password = ""
            home_screen = self.manager.get_screen("Home")
            home_screen.on_enter()
            if hasattr(home_screen, 'already_loaded'):
                home_screen.already_loaded = False

            self.deleted_acc.dismiss()
            self.manager.current = "Dashboard"
        
        except Exception as e:
            print("Error: ", e)
        
    # Cancel Delete Account
    def cancel_delete_account(self, obj):
        print("Cancel Delete Account")
        self.deleted_acc.dismiss()

    # Logout Account
    def logout_account(self):
        ok_btn = MDFlatButton(text="Yes", on_release=self.ok_logout_account)
        cancel_btn = MDFlatButton(text="Cancel", on_release=self.cancel_logout_account)
        self.logout_acc = MDDialog(title="Logout Account", text="Are your sure, you want to logout your account.", size_hint=(0.7, 0.2), buttons=[cancel_btn, ok_btn])
        self.logout_acc.open()

    # Ok Logout Account
    def ok_logout_account(self, obj):
        print("Logout Account")
        app = App.get_running_app()

        app.username = ""
        app.password = ""
        home_screen = self.manager.get_screen("Home")
        home_screen.on_enter()
        if hasattr(home_screen, 'already_loaded'):
            home_screen.already_loaded = False
            self.manager.current = "Dashboard"
            self.logout_acc.dismiss()
            print("Stopped")
    
    # Cancel Delete Account
    def cancel_logout_account(self, obj):
        print("Cancel Logout")
        self.logout_acc.dismiss()
