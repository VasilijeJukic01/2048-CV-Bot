import cv2
import numpy as np


class MoveRecorder:
    """Record gameplay frames with annotations to a video file.

        This class captures frames of the game screen, annotates them with the action taken
        ('up', 'down', 'left', 'right'), and saves the sequence as a video file.

        Attributes:
            output_path (str): The file path where the output video will be saved.
            fps (int): Frames per second for the output video. Default is 10.
            frames (List[np.ndarray]): A list to store the frames captured during recording.
            is_recording (bool): A flag indicating whether recording is currently active.
    """
    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        self.is_recording = True
        print("Recording started.")

    def record_frame(self, frame: np.ndarray, action: str):
        if not self.is_recording:
            return
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.frames.append(annotated_frame)

    def stop_recording(self):
        if not self.is_recording:
            return
        print("Stopping recording and writing video file.")
        try:
            height, width, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()
            print(f"Video file saved as {self.output_path}")
        except Exception as e:
            print(f"Error while writing video file: {e}")
        finally:
            self.is_recording = False
            self.frames = []
