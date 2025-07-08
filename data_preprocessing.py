import os
import shutil

# Đường dẫn tới thư mục chứa các folder của audio speech actors
source_root = r'C:\Users\ADMIN\Documents\INSA_Lyon\INSA 3A\SEMESTRE 2\Formation Générale\Audio-Emotion-Recognition\data\audio_speech_actors_01-24'
# Đường dẫn tới thư mục đích (data)
target_root = r'C:\Users\ADMIN\Documents\INSA_Lyon\INSA 3A\SEMESTRE 2\Formation Générale\Audio-Emotion-Recognition\data'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(target_root, exist_ok=True)

# Duyệt qua tất cả các folder trong source_root
for folder_name in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder_name)
    if os.path.isdir(folder_path):
        # Duyệt qua tất cả các file trong folder
        for file_name in os.listdir(folder_path):
            source_file = os.path.join(folder_path, file_name)
            if os.path.isfile(source_file):
                target_file = os.path.join(target_root + '\\' + folder_name, file_name)
                # Nếu file trùng tên, thêm hậu tố để tránh ghi đè
                base, ext = os.path.splitext(file_name)
                count = 1
                while os.path.exists(target_file):
                    target_file = os.path.join(target_root + '\\' + folder_name, f"{base}_{count}{ext}")
                    count += 1
                shutil.move(source_file, target_file)