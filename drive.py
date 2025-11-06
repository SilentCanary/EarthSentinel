# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 12:01:46 2025

@author: advit
"""


from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import zipfile
import subprocess


DRIVE_FOLDER = "HP_SAT_EXPORTS"

LOCAL_FOLDER = "downloaded_weeks"
os.makedirs(LOCAL_FOLDER, exist_ok=True)

print("🔑 Authenticating Google Drive...")
gauth = GoogleAuth()

gauth.LoadCredentialsFile("credentials.json")
gauth.settings["client_config_file"] = "client_secrets.json"



if gauth.credentials is None:
    print("⚠️ No stored credentials, doing OAuth login...")
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    print("♻️ Refreshing expired token...")
    gauth.Refresh()
else:
    print("✅ Using cached credentials...")

gauth.SaveCredentialsFile("credentials.json")
drive = GoogleDrive(gauth)

print("📂 Looking for Drive folder:", DRIVE_FOLDER)


folder_list = drive.ListFile({'q': "title='%s' and mimeType='application/vnd.google-apps.folder' and trashed=false" % DRIVE_FOLDER}).GetList()
if len(folder_list) == 0:
    raise ValueError("❌ Folder not found in Drive. Did GEE finish exporting?")
folder_id = folder_list[0]['id']

print("✅ Folder found. ID:", folder_id)
print("⬇ Downloading exported files...")

file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
print("Files found in Drive folder:")
for f in file_list:
    print(f['title'])

for f in file_list:
    filename = os.path.join(LOCAL_FOLDER, f['title'])
    if os.path.exists(filename):
        print(f"✅ Already downloaded → {filename}")
        continue
    print(f"Downloading → {filename}")
    f.GetContentFile(filename)

print("✅ All files downloaded!")

# ---- Extract any ZIPs ----
for file in os.listdir(LOCAL_FOLDER):
    if file.endswith(".zip"):
        zip_path = os.path.join(LOCAL_FOLDER, file)
        print("🗜 Extracting:", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_FOLDER)

print("✅ Extraction complete.")

print("🧩 Running patch creation (create_chunked_numpy_arrays.py)...")
subprocess.run(["python", "create_chunks.py"], check=True)


print("🤖 Running model inference (model_inference.py)...")
subprocess.run(["python", "inference.py"], check=True)


print("🎉 PIPELINE COMPLETE.")
