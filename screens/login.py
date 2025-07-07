from kivymd.uix.screen import MDScreen
from kivymd.uix.banner.banner import MDFlatButton
from kivymd.uix.dialog import MDDialog
import psycopg2
from kivy.app import App
import bcrypt
from screens.db import get_db_cursor

# Login
class Login(MDScreen):
    def login(self):
        self.usernameoremail = self.manager.get_screen("Login").ids.usernameoremail.text
        self.lpassword = self.manager.get_screen("Login").ids.lpassword.text

        self.hash_password = self.lpassword.encode('utf-8')

        app = App.get_running_app()
        app.username = self.usernameoremail
        app.password = self.hash_password


        ok_btn = MDFlatButton(text="OK", on_release=self.ok_dialog)
        
        # Show Not filled
        if not self.usernameoremail or not self.hash_password:
            self.logedin = MDDialog(title="Not Filled", text="Please fill all fields", size_hint=(0.7, 0.2), buttons=[ok_btn])
            self.logedin.open()
            self.ok = False
        
        # Database used
        else:
            try:
                # Check if the entered username/email exists in the database
                conn, cur = get_db_cursor()
                cur.execute("SELECT password FROM user_data WHERE username = %s OR email = %s", (self.usernameoremail, self.usernameoremail))
                user_pass = cur.fetchone()
                print(user_pass)

                # If the user exists
                if user_pass: 
                    # stored_password = user_pass[0]
                    # stored_password = user_pass[0]
                    # stored_password = user_pass[0].encode('latin1').decode('utf-8')
                    stored_password = user_pass[0].decode('utf-8') if isinstance(user_pass[0], bytes) else user_pass[0]

                    print("Stored Password :- ", stored_password)
                    print("Hashed Password :- ", self.hash_password)

                    # if stored_password == self.lpassword:
                    
                    if bcrypt.checkpw(self.hash_password, stored_password.encode("utf-8")):
                        self.manager.current = "Home"
                        self.logedin = MDDialog(title="Login Successful", text="Congratulations, Login Successful", size_hint=(0.7, 0.2), buttons=[ok_btn])
                        self.logedin.open()
                        self.manager.get_screen("Login").ids.usernameoremail.text = ""
                        self.manager.get_screen("Login").ids.lpassword.text = ""
                        self.ok = True
                    
                    # If Password is Incorrect
                    else:
                        self.logedin = MDDialog(title="Login Failed", text="Incorrect password", size_hint=(0.7, 0.2), buttons=[ok_btn])
                        self.logedin.open()
                        self.ok = False

                # Check if spaces are in password
                elif " " in self.lpassword:
                    self.logedin = MDDialog(title="Login Failed", text="Please don't add spaces in PASSWORD", size_hint=(0.7, 0.2), buttons=[ok_btn])
                    self.logedin.open()
                    self.ok = False

                elif " " in self.usernameoremail:
                    self.logedin = MDDialog(title="Login Failed", text="Please don't add spaces in USERNAME", size_hint=(0.7, 0.2), buttons=[ok_btn])
                    self.logedin.open()
                    self.ok = False

                # If user not found
                else:
                    self.logedin = MDDialog(title="Login Failed", text="User not found", size_hint=(0.7, 0.2), buttons=[ok_btn])
                    self.logedin.open()
                    self.ok = False

            except psycopg2.Error as err:
                self.logedin = MDDialog(title="Database Error", text=str(err), size_hint=(0.7, 0.2), buttons=[ok_btn])
                self.logedin.open()
                self.ok = False

            finally:
                # if conn.is_connected():
                cur.close()
                conn.close()

        print(f"Login Attempt: {self.usernameoremail}, {self.hash_password}")

    # Will remove the dialog box
    def ok_dialog(self, obj):
        if self.ok:
            self.logedin.dismiss()
        else:
            self.logedin.dismiss()
