from kivymd.uix.screen import MDScreen
from kivymd.uix.dialog import MDDialog
from kivymd.uix.banner.banner import MDFlatButton
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import bcrypt
from screens.db import get_db_cursor
import psycopg2
import smtplib

# Register
class Register(MDScreen):

    def registration_successful(self):
        to_email = self.manager.get_screen("Register").ids.email.text
        subject = "VisionChat App OTP"
        from_email = "visionchat405@gmail.com"
        app_password = "fhvfnwuoqnkmoitz" 

        try:
            with open("screens/otp_template.html", "r", encoding="utf-8") as file:
                html_template = file.read()

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = from_email
            msg["To"] = to_email
            msg.attach(MIMEText(html_template, "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(from_email, app_password)
                server.sendmail(from_email, to_email, msg.as_string())

            print("OTP sent successfully")

        except Exception as e:
            print("Error while sending email:", e)

    def register(self):
        self.firstname = self.manager.get_screen("Register").ids.firstname.text
        self.username = self.manager.get_screen("Register").ids.username.text
        self.email = self.manager.get_screen("Register").ids.email.text
        self.p = self.manager.get_screen("Register").ids.password.text


        try:
            print("------- Identifier --------") 

            conn, cur = get_db_cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_data (
                id SERIAL PRIMARY KEY,
                firstname VARCHAR(255),
                username VARCHAR(255) UNIQUE,
                email VARCHAR(255) UNIQUE,
                password VARCHAR(255),
                img_upload VARCHAR(255)
                );
            """)
            # Check if username or email already exists
            cur.execute("SELECT * FROM user_data WHERE username = %s OR email = %s", (self.username, self.email))
            user_exists = cur.fetchone()

            if user_exists:
                # User already exists
                self.ok_btn = MDFlatButton(text="OK", on_release=self.ok_dialog)
                self.db_error = MDDialog(
                    title="User Already Exists",
                    text="A user with this username or email already exists. Try another one.",
                    size_hint=(0.7, 0.2),
                    buttons=[self.ok_btn]
                )
                self.db_error.open()
                return  

            # Hash the password
            self.password = bcrypt.hashpw(self.p.encode('utf-8'), bcrypt.gensalt())

            print("Hashed Decode Password :- ", self.password)


            # Insert new user
            cur.execute("INSERT INTO user_data (firstname, username, email, password) VALUES (%s, %s, %s, %s)",
                        (self.firstname, self.username, self.email, self.password.decode('utf-8')))

            conn.commit()
            print("User registered successfully")

            # Show success dialog
            self.cancel_btn = MDFlatButton(text="Cancel", on_release=self.cancel_dialog)
            self.success_register = MDDialog(
                title="Registration Successful",
                text="Your registration is successful!",
                size_hint=(0.7, 0.2),
                buttons=[self.cancel_btn]
            )
            self.success_register.open()
            self.registration_successful()

            # Clear input fields
            self.manager.get_screen("Register").ids.firstname.text = ""
            self.manager.get_screen("Register").ids.username.text = ""
            self.manager.get_screen("Register").ids.email.text = ""
            self.manager.get_screen("Register").ids.password.text = ""

        except psycopg2.Error as err:
            print("Database Error:", err)
            self.ok_btn = MDFlatButton(text="OK", on_release=self.ok_dialog)
            self.db_error = MDDialog(
                title="Database Error",
                text=str(err),
                size_hint=(0.7, 0.2),
                buttons=[self.ok_btn]
            )
            self.db_error.open()

        finally:
            # if conn is not None and conn.is_connected():
            cur.close()
            conn.close()


    def ok_dialog(self, obj):
        self.db_error.dismiss()

    # def logedIn(self, obj):
    #     self.manager.current = "Login"
    #     self.success_register.dismiss()
    
    def cancel_dialog(self, obj):
        self.success_register.dismiss()