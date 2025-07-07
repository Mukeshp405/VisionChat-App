from kivymd.uix.screen import MDScreen
from kivymd.uix.banner.banner import MDFlatButton
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from kivymd.uix.dialog import MDDialog
import random
import smtplib
import re

# OTP Page
class OTP(MDScreen):
    # Generate OTP
    def generate_otp(self):
        try:
            """Generate a 6-digit OTP."""
            self.otp = str(random.randint(100000, 999999))
            return self.otp
        except Exception as e:
            print(e)

    def send_otp(self):
        self.otp_method()
        if not self.er:
            return

        to_email = self.manager.get_screen("Register").ids.email.text
        otp = self.generate_otp()
        subject = "VisionChat App OTP"
        from_email = "visionchat405@gmail.com"
        app_password = "fhvfnwuoqnkmoitz"  

        message_text = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #f9f9f9; padding: 20px;">
                <div style="max-width: 600px; margin: auto; background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <h2 style="color: orange;">Your OTP Code</h2>
                    <p style="font-size: 16px;">Use the following One-Time Password to complete your login or verification process:</p>
                    <div style="font-size: 24px; font-weight: bold; margin: 20px 0; color: #333;">{otp}</div>
                    <p style="font-size: 14px; color: #888;">Please do not share it with anyone.</p>
                </div>
            </body>
            </html>
        """
        # special_chars = "!@#$%^&*()_-+={}|\:<>:?[];',./`~"
        # if any(char in special_chars for char in self.username):
        #     self.notfilled = MDDialog(
        #         title="Invalid Username",
        #         text="Username cannot contain special characters.",
        #         size_hint=(0.7, 0.2),
        #         buttons=[self.ok_btn]
        #     )
        #     self.notfilled.open()
        #     self.er = True
        #     return
        # elif "@gmail.com" in to_email or not self.username or not self.password or not self.firstname:
        #     self.show_dialog("Not Filled2", "Please fill all columns")
        if "@gmail.com" in to_email and self.username and self.password and self.firstname:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = from_email
                msg["To"] = to_email
                msg.attach(MIMEText(message_text, "html"))

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(from_email, app_password)
                    server.sendmail(from_email, to_email, msg.as_string())
                    print("mail send .com", otp)
                    print(to_email)

                self.show_dialog("Success", f"OTP sent to {to_email}")
            except Exception as e:
                self.show_dialog("Error", f"Email failed: {e}")

    def show_dialog(self, title, text):
        dialog = MDDialog(title=title, text=text, size_hint=(0.7, 0.2))
        dialog.open()
    
    def otp_method(self):
        self.er = True
        self.firstname = self.manager.get_screen("Register").ids.firstname.text
        self.username = self.manager.get_screen("Register").ids.username.text
        self.email = self.manager.get_screen("Register").ids.email.text
        self.password = self.manager.get_screen("Register").ids.password.text

        self.ok_btn = MDFlatButton(text="OK", on_release=self.ok_dialog)

        # Check empty fields
        if not self.firstname or not self.username or not self.email or not self.password:
            self.notfilled = MDDialog(title="Not Filled", text="Please fill all columns", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.notfilled.open()
            self.er = True
            return

        # Check for spaces
        if " " in self.username:
            self.inv_email = MDDialog(title="Invalid Username", text="Please don't add spaces in USERNAME", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.inv_email.open()
            self.er = False
            return

        if " " in self.password:
            self.inv_email = MDDialog(title="Invalid Password", text="Please don't add spaces in PASSWORD", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.inv_email.open()
            self.er = False
            return

        if " " in self.email:
            self.inv_email = MDDialog(title="Invalid Email", text="Please don't add spaces in EMAIL", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.inv_email.open()
            self.er = False
            return

        if "@gmail.com" not in self.email:
            self.inv_email = MDDialog(title="Invalid Email", text="Please enter a valid email", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.inv_email.open()
            self.er = False
            return
        
        if len(self.username) < 5:
            self.inv_email = MDDialog(title="Invalid Username", text="Username must be at least 5 characters long.", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.inv_email.open()
            self.er = False
            return
        
        if len(self.password) < 8:
            self.inv_email = MDDialog(title="Invalid Password", text="Password must be at least 8 characters long.", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.inv_email.open()
            self.er = False
            return

        # Check for special characters in username
        if not re.match("^[A-Za-z0-9_]+$", self.username):
            self.inv_email = MDDialog(title="Invalid Username", text="Username cannot contain special characters.", size_hint=(0.7, 0.2), buttons=[self.ok_btn])
            self.inv_email.open()
            self.er = False
            return
        
         # If all checks pass, then go to OTP page
        self.manager.current = "OTP_Page"

    def ok_dialog(self, obj):
        if self.er:
            self.notfilled.dismiss()
        elif not self.er:
            self.inv_email.dismiss()

    def check_otp(self):    
        user_otp = self.manager.get_screen("OTP_Page").ids.enter_otp.text
        print("GOTP :- ", self.otp)
        print("User OTP :- ", user_otp)
        if self.otp == user_otp:
            self.manager.current = "Login"
            self.manager.get_screen("Register").register()
        else:
            self.otp_check = MDDialog(title="Wrong OTP", text="Please enter correct OTP.", size_hint=(0.7, 0.2), buttons=[MDFlatButton(text="OK", on_release=self.ok_check_dialog)])
            self.otp_check.open()
    
    def ok_check_dialog(self, obj):
        self.otp_check.dismiss()