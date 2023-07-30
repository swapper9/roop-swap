from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None
frame_count = None
frame_width = None
frame_height = None
fps = None
specified_fps = None
temp_frames_buffer = None
swap_face_frames = None
frame_processors: List[str] = []
keep_fps = None
keep_frames = None
skip_audio = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
max_memory_bytes = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
FACE_ENHANCER = None

SELECTED_FACE_DATA_INPUT = None
SELECTED_FACE_DATA_OUTPUT = None
