# IntelliSecureBank 🏦🔐

A **secure, intelligent banking system** that integrates **OCR KYC, Face Verification, Encrypted Storage, and Anomaly Detection**.  
Built for the **Samsung PRISM GenAI Hackathon 2025**, it ensures **robust security, user privacy, and real-time monitoring**.

---

## ✨ Features

- **🔑 Multi-Factor Admin Authentication**
  - Passwords hashed with **bcrypt**
  - **TOTP (Google Authenticator/Authy)** login
  - OTP delivery via **Twilio SMS** or **Email**

- **🧾 KYC Verification**
  - OCR (Tesseract) to extract details from Aadhaar/ID
  - Fuzzy matching for name validation

- **🖼 Face Verification**
  - Passport photo upload + **OpenCV DNN** live webcam match
  - Snapshot storage for audit/compliance

- **💳 Banking Operations**
  - Create, modify, delete accounts
  - Deposit/Withdraw with validation
  - Balance enquiry & search by account/name
  - Transaction history logging (`transactions.csv`)

- **🚨 Anomaly Detection**
  - Train **IsolationForest model** on transactions
  - Flag suspicious anomalies
  - Save flagged results to `transactions_flagged.csv`

- **📊 Data Export**
  - Encrypted account storage (`accounts.csv`)
  - Excel export with password-protection

---

## 🛠 Tech Stack

- **Language**: Python 3.10+  
- **Core Libraries**:  
  - `opencv-python`, `pytesseract`, `Pillow`, `numpy`  
  - `pandas`, `scikit-learn`  
  - `bcrypt`, `pyotp`, `python-dotenv`  
  - `yagmail`, `twilio`  
  - `xlsxwriter`, `win32com.client`  
- **Database**: SQLite (`admins.db`)  
- **Encryption**: AES/Fernet (`crypto_utils.py`)

---

## 📂 Project Structure


{repo_dir}/
├─ intell.py
├─ db.py
├─ crypto_utils.py
├─ admins.db
├─ accounts.csv
├─ transactions.csv
├─ requirements.txt
├─ README.md
├─ TeamName.pdf (or TeamName.md)
├─ models/
│  ├─ deploy.prototxt
│  ├─ res10_300x300_ssd_iter_140000.caffemodel
│  └─ nn4.small2.v1.t7
├─ aadhaarcards/
├─ passport_size_photos/
├─ admin_images/
├─ backups/
└─ logs/


## 🛠 Setup
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python db.py create-admin admin StrongPass!123 --role Admin
python intell.py

