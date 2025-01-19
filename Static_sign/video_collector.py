import cv2
import os
import json
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)

class SignDataCollector:
    def __init__(self):
        try:
            self.base_dir = "sign_language_data"
            self.metadata_file = "metadata.json"
            self.setup_directories()
            self.load_metadata()
        except Exception as e:
            logging.error("Error initializing SignDataCollector")
            logging.error(traceback.format_exc())
            raise e
        
    def setup_directories(self):
        try:
            os.makedirs(self.base_dir, exist_ok=True)
        except Exception as e:
            logging.error("Error setting up directories")
            logging.error(traceback.format_exc())
            raise e
        
    def load_metadata(self):
        try:
            self.metadata_path = os.path.join(self.base_dir, self.metadata_file)
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    'signs': {},
                    'total_recordings': 0,
                    'last_updated': None
                }
        except Exception as e:
            logging.error("Error loading metadata")
            logging.error(traceback.format_exc())
            raise e
            
    def save_metadata(self):
        try:
            self.metadata['last_updated'] = datetime.now().isoformat()
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=4)
        except Exception as e:
            logging.error("Error saving metadata")
            logging.error(traceback.format_exc())
            raise e
            
    def collect_sign(self, sign_name, signer_id, hand='right'):
        try:
            sign_dir = os.path.join(self.base_dir, sign_name)
            os.makedirs(sign_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Error: Could not open webcam.")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30
            
            frames_buffer = []
            
            recording = False
            frames_recorded = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to grab frame.")
                    break
                    
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                
                if recording:
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                    frames_recorded += 1
                    frames_buffer.append(frame.copy())
                    cv2.putText(display_frame, f"Frames: {frames_recorded}", 
                               (width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, "R: Record/Stop", (10, height-90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "S: Save Recording", (10, height-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Q: Quit", (10, height-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if frames_recorded > 0:
                    cv2.putText(display_frame, f"Saved: {saved_count}", (width-150, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Sign Language Data Collection', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):
                    if not recording:
                        recording = True
                        frames_buffer = []
                        frames_recorded = 0
                        logging.info("Recording started...")
                    else:
                        recording = False
                        logging.info("Recording stopped. Press 'S' to save or 'R' to record again.")
                        
                elif key == ord('s') and not recording and frames_recorded > 0:
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_filename = f"{sign_name}_{signer_id}_{hand}_{timestamp}.mp4"
                        video_path = os.path.join(sign_dir, video_filename)
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                        for buffered_frame in frames_buffer:
                            out.write(buffered_frame)
                        out.release()
                        
                        if sign_name not in self.metadata['signs']:
                            self.metadata['signs'][sign_name] = {
                                'total_recordings': 0,
                                'recordings': []
                            }
                            
                        recording_metadata = {
                            'filename': video_filename,
                            'signer_id': signer_id,
                            'hand': hand,
                            'timestamp': timestamp,
                            'frames': frames_recorded,
                            'duration': frames_recorded/fps
                        }
                        
                        self.metadata['signs'][sign_name]['recordings'].append(recording_metadata)
                        self.metadata['signs'][sign_name]['total_recordings'] += 1
                        self.metadata['total_recordings'] += 1
                        self.save_metadata()
                        
                        saved_count += 1
                        logging.info(f"Recording saved to {video_path}")
                        frames_recorded = 0
                        frames_buffer = []
                        
                    except Exception as e:
                        logging.error("Error saving recording")
                        logging.error(traceback.format_exc())
                        
                elif key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
            return saved_count
        except Exception as e:
            logging.error("Error during sign collection")
            logging.error(traceback.format_exc())
            return 0

def main():
    try:
        root = tk.Tk()
        root.withdraw()
        
        collector = SignDataCollector()
        
        while True:
            sign_name = simpledialog.askstring("Input", "Enter the sign to record (or 'quit' to exit):")
            if not sign_name or sign_name.lower() == 'quit':
                break
                
            signer_id = simpledialog.askstring("Input", "Enter signer ID:")
            if not signer_id:
                break
                
            hand = simpledialog.askstring("Input", "Enter hand (right/left):", 
                                        initialvalue="right")
            if hand not in ['right', 'left']:
                hand = 'right'
                
            # Record the sign
            saved_count = collector.collect_sign(sign_name, signer_id, hand)
            
            if saved_count > 0:
                messagebox.showinfo("Success", f"Saved {saved_count} recordings")
            
            # Ask to continue
            if not messagebox.askyesno("Continue?", "Record another sign?"):
                break
                
        logging.info("Data collection completed.")
        logging.info(f"Total recordings: {collector.metadata['total_recordings']}")
    except Exception as e:
        logging.error("Error in main function")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
