import os
import cv2

def extract_limited_frames(input_folder, output_folder, target_frames=25):
    # Ekstensi video yang didukung
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder output dibuat: {output_folder}")

    files = os.listdir(input_folder)
    video_files = [f for f in files if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"Tidak ada file video yang ditemukan di folder: {input_folder}")
        return

    print(f"Menemukan {len(video_files)} video. Memulai ekstraksi (Target: ~{target_frames} frame per video)...")
    total_frames_saved = 0

    for video_name in video_files:
        video_path = os.path.join(input_folder, video_name)
        video_name_clean = os.path.splitext(video_name)[0]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Gagal membuka video: {video_name}")
            continue

        # Hitung total frame asli yang ada di video
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Hitung interval/jarak lompatan frame agar tersebar merata
        # Jika total frame lebih kecil dari target, ambil semua frame yang ada
        interval = max(1, total_video_frames // target_frames)
        
        print(f"Memproses: {video_name} (Total frame asli: {total_video_frames}, Ambil setiap kelipatan {interval} frame)")
        
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Hanya simpan frame yang masuk dalam hitungan kelipatan interval
            if frame_count % interval == 0 and saved_count < target_frames:
                frame_filename = f"{video_name_clean}_frame_{saved_count:03d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                total_frames_saved += 1
            
            frame_count += 1

        cap.release()
        print(f"Selesai! Berhasil mengambil {saved_count} frame dari {video_name}")

    print(f"\nProses selesai! Total {total_frames_saved} gambar disimpan di '{output_folder}'.")

# --- Pengaturan Folder & Target ---
FOLDER_INPUT = "videos" 
FOLDER_OUTPUT = "dataset_frames_limited"
TARGET_FRAME = 25 # Anda bisa mengubah ini ke angka antara 20-30 sesuai kebutuhan

if __name__ == "__main__":
    extract_limited_frames(FOLDER_INPUT, FOLDER_OUTPUT, TARGET_FRAME)
