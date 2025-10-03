from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import base64
import os

BLOCK_SIZE = 16
KEY_SIZE = 32
PADDING = b' '

def generate_key(key_file='aes.key'):
    key = get_random_bytes(KEY_SIZE)
    with open(key_file, 'wb') as f:
        f.write(key)
    print("🔐 AES-256 key saved to", key_file)

def load_key(key_file='aes.key'):
    with open(key_file, 'rb') as f:
        return f.read()

def pad(data):
    return data + (BLOCK_SIZE - len(data) % BLOCK_SIZE) * PADDING

def encrypt_text(plaintext, key):
    iv = get_random_bytes(BLOCK_SIZE)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_text = pad(plaintext.encode())
    encrypted = cipher.encrypt(padded_text)
    return base64.b64encode(iv + encrypted).decode()

def decrypt_text(ciphertext, key):
    raw = base64.b64decode(ciphertext)
    iv = raw[:BLOCK_SIZE]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(raw[BLOCK_SIZE:])
    return decrypted.rstrip(PADDING).decode()
