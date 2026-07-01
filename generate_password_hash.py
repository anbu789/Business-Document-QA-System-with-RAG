"""
generate_password_hash.py
--------------------------
Run this locally to generate a bcrypt hash for a new user's password,
then paste the hash into auth_config.yaml (never the plaintext password).

Usage:
    python generate_password_hash.py
"""

import getpass
import bcrypt

if __name__ == "__main__":
    password = getpass.getpass("Enter the password to hash: ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("Passwords did not match. Try again.")
    else:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        print("\nPaste this into auth_config.yaml as the user's `password:` value:\n")
        print(hashed)
