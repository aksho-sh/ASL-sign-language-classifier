import cv2
import os
from tqdm import tqdm
import shutil

class VideoProcessor:
    def __init__(self, target_height=480):
        self.target_height = target_height

    def get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        return {"width": width, "height": height, "fps": fps}

    def resize_video(self, input_path, output_path):
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error opening video: {input_path}")
                return False
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            aspect_ratio = width / height
            new_height = self.target_height
            new_width = int(new_height * aspect_ratio)
            new_width = new_width + (new_width % 2)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                resized_frame = cv2.resize(frame, (new_width, new_height), 
                                         interpolation=cv2.INTER_AREA)
                out.write(resized_frame)

            cap.release()
            out.release()
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            return False

def process_dataset(input_base_dir, output_base_dir):
    subdirs = ['hello', 'negative', 'thank_you']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_base_dir, subdir), exist_ok=True)

    processor = VideoProcessor(target_height=480)
    stats = {subdir: {'processed': 0, 'copied': 0, 'failed': 0} for subdir in subdirs}

    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        file_count = 1

        video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        
        print(f"\nProcessing {subdir} directory ({len(video_files)} videos)...")
        
        for video_file in tqdm(video_files, desc=f"Processing {subdir}"):
            input_path = os.path.join(input_dir, video_file)
            new_filename = f"{file_count}.mp4"
            output_path = os.path.join(output_dir, new_filename)

            info = processor.get_video_info(input_path)
            if info is None:
                stats[subdir]['failed'] += 1
                continue

            if info['height'] > 480:
                if processor.resize_video(input_path, output_path):
                    stats[subdir]['processed'] += 1
                    file_count += 1
                else:
                    stats[subdir]['failed'] += 1
            else:
                try:
                    shutil.copy2(input_path, output_path)
                    stats[subdir]['copied'] += 1
                    file_count += 1
                except Exception as e:
                    print(f"Error copying {input_path}: {str(e)}")
                    stats[subdir]['failed'] += 1

    return stats

def main():
    input_base_dir = "dataset"
    output_base_dir = "processed_dataset"
    
    print("Starting dataset processing...")
    stats = process_dataset(input_base_dir, output_base_dir)
    
    print("\nProcessing Summary:")
    for subdir, counts in stats.items():
        print(f"\n{subdir} directory:")
        print(f"  Processed (resized): {counts['processed']}")
        print(f"  Copied (no resize): {counts['copied']}")
        print(f"  Failed: {counts['failed']}")
        print(f"  Total successful: {counts['processed'] + counts['copied']}")

if __name__ == "__main__":
    main()