import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created output folder: {folder_path}")
    else:
        print(f"Output folder already exists: {folder_path}")

def extract_color_histogram(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def match_to_memory(current_feature, memory, threshold=0.3):
    best_match_id = None
    best_score = float('inf')
    for mid, mem_feature in memory.items():
        score = cv2.compareHist(mem_feature, current_feature, cv2.HISTCMP_BHATTACHARYYA)
        if score < best_score and score < threshold:
            best_score = score
            best_match_id = mid
    return best_match_id

def process_video(input_video_path, output_folder, model_path):
    ensure_folder_exists(output_folder)

    model = YOLO(model_path)
    print(model.names)
    tracker = DeepSort(max_age=30, nn_budget=100)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_path = os.path.join(output_folder, f"{video_name}_tracked.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Player appearance memory: id -> feature
    reid_memory = {}

    # Maintain global ID count
    global_id_map = {}
    next_global_id = 1

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []
        crops = []

        for result in results:
            player_class_id = [k for k, v in model.names.items() if v.lower() == 'player']
            if not player_class_id:
                raise ValueError("No 'player' class found in model.")
            player_class_id = player_class_id[0]

            for box in result.boxes:
                class_id=int(box.cls)
                if class_id in [1, 2, 3] and box.conf > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    player_crop = frame[y1:y2, x1:x2]
                    if player_crop.size == 0:
                        continue
                    feature = extract_color_histogram(player_crop)
                    detections.append((bbox, conf, model.names[class_id]))
                    crops.append((bbox, feature))

        # Track
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            try:
                track_id = int(track.track_id)
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Extract current feature for re-ID
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                current_feature = extract_color_histogram(crop)

                # Match to memory
                if track_id not in global_id_map:
                    matched_id = match_to_memory(current_feature, reid_memory)
                    if matched_id:
                        global_id_map[track_id] = matched_id
                    else:
                        global_id_map[track_id] = next_global_id
                        reid_memory[next_global_id] = current_feature
                        next_global_id += 1

                global_id = global_id_map[track_id]
                color = (
                    int((global_id * 50) % 256),
                    int((global_id * 80) % 256),
                    int((global_id * 110) % 256)
                )

                # Draw box and global ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {global_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                print(f"Error drawing track: {e}")
                continue

        out.write(frame)
        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end='\r')

    cap.release()
    out.release()
    print(f"\n✅ Done. Output saved to: {output_path}")

if __name__ == "__main__":
    INPUT_VIDEO = "15sec_input_720p.mp4"
    OUTPUT_FOLDER = "output"
    MODEL_PATH = "best.pt"

    process_video(INPUT_VIDEO, OUTPUT_FOLDER, MODEL_PATH)
