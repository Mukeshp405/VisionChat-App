from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
from datetime import datetime
from screens.db import get_db_cursor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Set up SocketIO
socketio = SocketIO(app, cors_allowed_origins='*')

connected_users = {}

def get_user_info_by_username(username):
    try:
        conn, cur = get_db_cursor()
        cur.execute("SELECT id, username FROM user_data WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()
        return user 
    except Exception as e:
        print("Error:", e)
        return None

def save_message_to_db(sender_id, sender_name, receiver_id, receiver_name, message, time, is_seen=False):
    try:
        conn, cur = get_db_cursor()
        cur.execute("""
            INSERT INTO chat_data (
                sender_user_id, sender_user_name,
                receiver_user_id, receiver_user_name,
                message, is_seen, timestamp
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (sender_id, sender_name, receiver_id, receiver_name, message, is_seen, datetime.fromisoformat(time)))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("Database error:", e)

# Socket.IO Events
@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('join')
def handle_join(data):
    username = data.get('username')
    connected_users[username] = request.sid
    print(f"{username} joined with session ID {request.sid}")

@socketio.on('private_message')
def handle_private_message(data):
    sender_username = data.get('from')
    receiver_username = data.get('to')
    message = data.get('message')
    time = data.get('time')

    print(f"Private message from {sender_username} to {receiver_username}: {message}")

    sender = get_user_info_by_username(sender_username)
    receiver = get_user_info_by_username(receiver_username)

    if not sender or not receiver:
        print("Sender or receiver user not found in DB")
        return

    sender_id, sender_name = sender
    receiver_id, receiver_name = receiver

    # Save message to DB
    save_message_to_db(sender_id, sender_name, receiver_id, receiver_name, message, time)

    if receiver_username in connected_users:
        emit('new_private_message', {
            'from': sender_name,
            'message': message,
            'time': time
        }, room=connected_users[receiver_username])

@socketio.on('disconnect')
def handle_disconnect():
    for username, sid in list(connected_users.items()):
        if sid == request.sid:
            del connected_users[username]
            print(f"{username} disconnected")
            break

# Run server
if __name__ == '__main__':
    print("Socket.IO server running on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
