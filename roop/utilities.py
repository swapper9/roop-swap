import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib
import cv2
import numpy as np

from pathlib import Path
from typing import List, Any
from tqdm import tqdm
from scipy.spatial import distance

import roop.globals

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of',
               'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30.0


def extract_frames(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(['-i', target_path, '-pix_fmt', 'rgb24', os.path.join(temp_directory_path, '%04d.png')])


def create_video(target_path: str, fps: float = 30.0) -> None:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(['-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.png'), '-c:v', roop.globals.video_encoder,
                '-crf', str(roop.globals.video_quality), '-pix_fmt', 'yuv420p', '-vf',
                'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
    # ffmpeg -hide_banner -hwaccel auto -loglevel error -r 30.0 -i G:/delme\\temp\\te1533...0\\%04d.png -c:v libx264 -crf 18


def write_video(target_path: str, part: int, parts: int, fps: float = 30.0) -> None:
    if parts > 1:
        temp_output_path = get_temp_part_output_path(target_path, str(part))
    else:
        temp_output_path = get_temp_output_path(target_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(temp_output_path, fourcc, fps, (roop.globals.frame_width, roop.globals.frame_height))
    for frame in range(0, len(roop.globals.temp_frames_buffer)):
        video.write(roop.globals.temp_frames_buffer[frame])
    video.release()


def concatenate_temp_video_parts(target_path: str, fps: float, *parts: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    video = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                            (roop.globals.frame_width, roop.globals.frame_height))

    for part in parts:
        temp_part_path = get_temp_part_output_path(target_path, str(part))
        curr_v = cv2.VideoCapture(temp_part_path)
        while curr_v.isOpened():
            r, frame = curr_v.read()
            if not r:
                break
            video.write(frame)
    video.release()


def split_video() -> (int, int):
    video = cv2.VideoCapture(roop.globals.target_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, 1)
    res, frame_buffer = video.read()
    frame_size = frame_buffer.nbytes
    total_frame_size = frame_size * roop.globals.frame_count
    parts = total_frame_size // roop.globals.max_memory_bytes + 1
    frames_in_part = roop.globals.max_memory_bytes // frame_size
    video.release()
    return parts, frames_in_part


def extract_frames_to_buffer(begin_frame: int = 0, end_frame: int = roop.globals.frame_count) -> None:
    frame_range = end_frame - begin_frame
    roop.globals.temp_frames_buffer = np.empty((frame_range, roop.globals.frame_height, roop.globals.frame_width, 3),
                                               np.dtype('uint8'))
    fc = 0
    ret = True
    cap = cv2.VideoCapture(roop.globals.target_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin_frame)
    while fc < frame_range and ret:
        ret, roop.globals.temp_frames_buffer[fc] = cap.read()
        fc += 1
    cap.release()


def read_source_params() -> None:
    if not roop.globals.specified_fps:
        roop.globals.fps = detect_fps(roop.globals.target_path)
    cap = cv2.VideoCapture(roop.globals.target_path)
    roop.globals.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roop.globals.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roop.globals.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()


def get_temp_part_output_path(target_path: str, part: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, 'temp_' + part + '.mp4')


def from_path_to_array_index(path: str) -> int:
    return int(path[-10:-4])


def from_array_index_to_path(ind: int) -> str:
    return get_temp_directory_path(roop.globals.target_path) + str(ind).zfill(6) + '.png'


def get_virtual_paths_list(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    frame_indices = list(range(0, len(roop.globals.temp_frames_buffer)))
    path_list = []
    for idx in frame_indices:
        path_list.append(temp_directory_path + '/' + from_array_index_to_path(idx))
    return path_list


def create_virtual_paths_list() -> List[str]:
    paths = []
    for frame_idx in roop.globals.swap_face_frames:
        paths.append(from_array_index_to_path(frame_idx))
    return paths


def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        ['-i', temp_output_path, '-i', target_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y',
         output_path])
    if not done:
        move_temp(target_path, output_path)


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.png')))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path


def get_destfilename_from_path(srcfilepath: str, destfilepath: str, extension: str) -> str:
    fn = os.path.splitext(os.path.basename(srcfilepath))[0]
    return os.path.join(destfilepath, f'{fn}{extension}')


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc=f'Downloading {url}', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path,
                                           reporthook=lambda count, block_size, total_size: progress.update(
                                               block_size))  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'
    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'
    return 'cpu'


# Taken from https://stackoverflow.com/a/68842705
def get_platform():
    if sys.platform == 'linux':
        try:
            proc_version = open('/proc/version').read()
            if 'Microsoft' in proc_version:
                return 'wsl'
        except:
            pass
    return sys.platform


def open_with_default_app(filename):
    if filename == None:
        return
    platform = get_platform()
    if platform == 'darwin':
        subprocess.call(('open', filename))
    elif platform in ['win64', 'win32']:
        os.startfile(filename.replace('/', '\\'))
    elif platform == 'wsl':
        subprocess.call('cmd.exe /C start'.split() + [filename])
    else:  # linux variants
        subprocess.call(('xdg-open', filename))


def compute_cosine_distance(emb1, emb2):
    return distance.cosine(emb1, emb2)
