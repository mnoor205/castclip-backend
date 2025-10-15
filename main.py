import modal
from fastapi import Depends, status, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import os
import uuid
import pathlib
import boto3
import whisperx
import subprocess
import time
import json
from google import genai
import shutil
import pickle
import glob
import numpy as np
from tqdm import tqdm
import cv2
import ffmpegcv
from google.genai import types
from yt_dlp import YoutubeDL
import httpx
import concurrent.futures
import requests
import logging
from typing import Optional
from fastapi.responses import JSONResponse

os.environ.setdefault("PYANNOTE_AUDIO_DISABLE_CUDA", "True")




class ProcessVideoRequest(BaseModel):
    s3_key: str
    user_id: str
    project_id: str
    clip_count: int = 1
    style: int = 1


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn9-cuda-12", "libcudnn9-dev-cuda-12"])
    .env({
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
    })
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("LR-ASD", "/LR-ASD", copy=True)
    .add_local_file("cookies.txt", "/cookies.txt"))

app = modal.App("ai-podcast-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clipper-modal-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()


def call_styling_endpoint(
    user_id: str,
    project_id: str,
    clip_id: str,
    s3_key: str,
    transcript: list,
    hook: str,
    caption_style_id: int,
    start_time: int,
    end_time: int,
    clip_number: int = 1,
    total_clips: int = 1
) -> tuple[bool, Optional[str], dict]:
    """
    Call the styling endpoint to generate styled video.
    The styling service processes asynchronously and sends a webhook when complete.
    
    Returns:
        (success: bool, error_message: Optional[str], payload: dict)
    """
    styling_url = "https://gnerate-videos-320842415829.us-south1.run.app/generate"
    
    # Default style options
    payload = {
        "userId": user_id,
        "projectId": project_id,
        "clipId": clip_id,
        "s3Key": s3_key,
        "transcript": transcript,
        "hook": hook,
        "hookStyle": {
            "fontSize": 75.7831885551744,
            "position": {
                "x": 50.74156065244932,
                "y": 14.779669015903702
            }
        },
        "captionsStyle": {
            "fontSize": 97.56151895220424,
            "position": {
                "x": 50.134697991448455,
                "y": 76.50111739702355
            }
        },
        "captionStyleId": caption_style_id,
        "type": "generate",
        "start": start_time,
        "end": end_time,
        "clipIndex": clip_number,
        "totalClips": total_clips,
        "isLastClip": clip_number == total_clips
    }
    
    try:
        print(f"Sending clip {clip_id} to styling service...")
        response = requests.post(
            styling_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10  # Just need to confirm request was accepted
        )
        
        if response.status_code in [200, 202]:
            print(f"✅ Successfully sent clip {clip_id} to styling service")
            return (True, None, payload)
        else:
            error_msg = f"Styling endpoint returned {response.status_code}: {response.text[:200]}"
            print(f"❌ {error_msg}")
            return (False, error_msg, payload)
            
    except requests.exceptions.Timeout:
        # Timeout is a real error - styling service should respond quickly to accept the request
        error_msg = f"Styling endpoint timeout after 10s for clip {clip_id}"
        print(f"❌ {error_msg}")
        return (False, error_msg, payload)
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error calling styling endpoint: {str(e)}"
        print(f"❌ {error_msg}")
        return (False, error_msg, payload)
    except Exception as e:
        error_msg = f"Unexpected error calling styling endpoint: {str(e)}"
        print(f"❌ {error_msg}")
        return (False, error_msg, payload)


def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, framerate=25):
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()

    faces = [[] for _ in range(len(flist))]

    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)

            if 0 <= frame < len(faces):
                faces[frame].append(
                    {'track': tidx, 'score': avg_score, 's': track['proc_track']["s"][fidx],
                        'x': track['proc_track']["x"][fidx], 'y': track['proc_track']["y"][fidx]}
                )

    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")

    vout = None
    for fidx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces[fidx]

        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            vout = ffmpegcv.VideoWriterNV(
                file=temp_video_path,
                codec=None,
                fps=framerate,
                resize=(target_width, target_height)
            )

        # Determine processing mode based on face detection
        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            # Resize mode: Create blurred background with centered video
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(
                img, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            # Create blurred background
            scale_for_bg = max(
                target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_height = int(img.shape[0] * scale_for_bg)

            blurred_background = cv2.resize(img, (bg_width, bg_height))
            blurred_background = cv2.GaussianBlur(
                blurred_background, (61, 61), 0)

            # Crop blurred background to target size
            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_height - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y +
                                                    target_height, crop_x:crop_x + target_width]

            # Center the resized video on the blurred background
            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y +
                               resized_height, :] = resized_image

            vout.write(blurred_background)

        elif mode == "crop":
            # Crop mode: Focus on the detected face (existing logic)
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            center_x = int(
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(min(center_x - target_width // 2,
                        frame_width - target_width), 0)

            image_cropped = resized_image[0:target_height,
                                          top_x:top_x + target_width]

            vout.write(image_cropped)

    if vout:
        vout.release()

    try:
        ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                          f"-c:v copy -c:a aac -b:a 128k -movflags +faststart "
                          f"{output_path}")
        subprocess.run(ffmpeg_command, shell=True, check=True, text=True)
    except subprocess.CalledProcessError:
        # Fallback to re-encode video if stream copy fails
        ffmpeg_fallback = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                           f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k -movflags +faststart "
                           f"{output_path}")
        subprocess.run(ffmpeg_fallback, shell=True, check=True, text=True)




def compute_faces_by_frame(tracks, scores, num_frames: int):
    faces = [[] for _ in range(num_frames)]
    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        frames_list = track["track"]["frame"].tolist()
        for fidx, frame in enumerate(frames_list):
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice) if len(score_slice) > 0 else 0)
            if 0 <= frame < len(faces):
                faces[frame].append({
                    'track': tidx,
                    'score': avg_score,
                    's': track['proc_track']["s"][fidx],
                    'x': track['proc_track']["x"][fidx],
                    'y': track['proc_track']["y"][fidx]
                })
    return faces


def create_vertical_video_segment(faces_by_frame, pyframes_path: str, output_video_only_path: str, frame_start: int, frame_end: int, framerate: int = 25):
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()

    vout = None
    for frame_index in tqdm(range(frame_start, min(frame_end, len(flist))), total=max(0, frame_end - frame_start), desc="Rendering clip from batch frames"):
        fname = flist[frame_index]
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces_by_frame[frame_index] if frame_index < len(faces_by_frame) else []
        max_score_face = max(current_faces, key=lambda face: face['score']) if current_faces else None
        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            vout = ffmpegcv.VideoWriterNV(
                file=output_video_only_path,
                codec=None,
                fps=framerate,
                resize=(target_width, target_height)
            )

        if max_score_face:
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]
            center_x = int(max_score_face["x"] * scale)
            top_x = max(min(center_x - target_width // 2, frame_width - target_width), 0)
            image_cropped = resized_image[0:target_height, top_x:top_x + target_width]
            vout.write(image_cropped)
        else:
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(img, (target_width, resized_height), interpolation=cv2.INTER_AREA)
            scale_for_bg = max(target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_height = int(img.shape[0] * scale_for_bg)
            blurred_background = cv2.resize(img, (bg_width, bg_height))
            blurred_background = cv2.GaussianBlur(blurred_background, (61, 61), 0)
            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_height - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y + resized_height, :] = resized_image
            vout.write(blurred_background)

    if vout:
        vout.release()


def build_batch_video(base_dir: pathlib.Path, original_video_path: str, clip_args_list: list[dict]) -> tuple[str, pathlib.Path, list[float]]:
    batch_name = "batch_input"
    batch_video_path = base_dir / f"{batch_name}.mp4"

    segments_dir = base_dir / "batch_segments"
    segments_dir.mkdir(exist_ok=True)

    offsets = []
    current_offset = 0.0
    list_file_path = base_dir / "batch_concat_list.txt"

    with open(list_file_path, "w", encoding="utf-8") as lf:
        for idx, args in enumerate(clip_args_list):
            start = args["start"]
            end = args["end"]
            duration = max(end - start, 0)
            seg_path = segments_dir / f"seg_{idx:03d}.mp4"
            cut_cmd = (f"ffmpeg -y -i {original_video_path} -ss {start} -t {duration} -c copy {seg_path}")
            subprocess.run(cut_cmd, shell=True, check=True, capture_output=True, text=True)
            lf.write(f"file '{seg_path}'\n")
            offsets.append(current_offset)
            current_offset += duration

    concat_cmd = (f"ffmpeg -y -f concat -safe 0 -i {list_file_path} -c copy {batch_video_path}")
    subprocess.run(concat_cmd, shell=True, check=True, capture_output=True, text=True)

    return batch_name, batch_video_path, offsets


def generate_raw_clips_batched(base_dir: pathlib.Path, original_video_path: str, s3_key: str, clip_args_list: list[dict]) -> list[dict]:
    try:
        batch_name, batch_video_path, offsets = build_batch_video(base_dir, original_video_path, clip_args_list)

        columbia_command = (f"python Columbia_test.py --videoName {batch_name} "
                            f"--videoFolder {str(base_dir)} "
                            f"--pretrainModel weight/finetuning_TalkSet.model")
        subprocess.run(columbia_command, cwd="/LR-ASD", shell=True, check=True)

        batch_dir = base_dir / batch_name
        tracks_path = batch_dir / "pywork" / "tracks.pckl"
        scores_path = batch_dir / "pywork" / "scores.pckl"
        pyframes_path = batch_dir / "pyframes"
        pyavi_path = batch_dir / "pyavi"

        if not tracks_path.exists() or not scores_path.exists():
            raise FileNotFoundError("Batch tracks or scores not found")

        with open(tracks_path, "rb") as f:
            tracks = pickle.load(f)
        with open(scores_path, "rb") as f:
            scores = pickle.load(f)

        flist = glob.glob(os.path.join(str(pyframes_path), "*.jpg"))
        flist.sort()
        framerate = 25

        faces_by_frame = compute_faces_by_frame(tracks, scores, len(flist))

        results = []
        max_workers = min(len(clip_args_list), 3)

        def render_one(idx_and_args):
            idx, args = idx_and_args
            clip_id = str(uuid.uuid4())
            try:
                start = args["start"]
                end = args["end"]
                duration = max(end - start, 0)
                batch_offset = offsets[idx]
                frame_start = int(batch_offset * framerate)
                frame_end = int((batch_offset + duration) * framerate)

                clip_name = f"clip_{args['index']}_raw"
                clip_dir = base_dir / clip_name
                clip_dir.mkdir(parents=True, exist_ok=True)
                (clip_dir / "pywork").mkdir(exist_ok=True)
                pyavi_out = clip_dir / "pyavi"
                pyavi_out.mkdir(exist_ok=True)
                audio_path = pyavi_out / "audio.wav"
                video_only_path = pyavi_out / "video_only.mp4"
                vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"

                extract_cmd = f"ffmpeg -y -ss {start} -t {duration} -i {original_video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
                subprocess.run(extract_cmd, shell=True, check=True, capture_output=True, text=True)

                create_vertical_video_segment(
                    faces_by_frame=faces_by_frame,
                    pyframes_path=str(pyframes_path),
                    output_video_only_path=str(video_only_path),
                    frame_start=frame_start,
                    frame_end=frame_end,
                    framerate=framerate,
                )

                try:
                    ffmpeg_command = (f"ffmpeg -y -i {video_only_path} -i {audio_path} "
                                      f"-c:v copy -c:a aac -b:a 128k -movflags +faststart "
                                      f"{vertical_mp4_path}")
                    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)
                except subprocess.CalledProcessError:
                    ffmpeg_fallback = (f"ffmpeg -y -i {video_only_path} -i {audio_path} "
                                       f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k -movflags +faststart "
                                       f"{vertical_mp4_path}")
                    subprocess.run(ffmpeg_fallback, shell=True, check=True, text=True)

                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                    endpoint_url=os.environ["R2_ENDPOINT_URL"],
                    region_name="auto"
                )

                s3_key_dir = os.path.dirname(s3_key)
                output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
                s3_client.upload_file(
                    str(vertical_mp4_path), "ai-podcast-clipper", output_s3_key,
                    ExtraArgs={'ContentType': 'video/mp4'}
                )

                raw_clip_url = f"https://castclip.revolt-ai.com/{output_s3_key}"

                return {
                    "index": args["index"],
                    "status": "success",
                    "clip_id": clip_id,
                    "raw_clip_url": raw_clip_url,
                    "s3_key": output_s3_key,
                    "transcript_segments": args["transcript_segments"],
                    "hook": args["hook"],
                    "start": args["start"],
                    "end": args["end"]
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {
                    "index": args["index"],
                    "status": "error",
                    "error": str(e),
                    "clip_id": clip_id
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for res in executor.map(render_one, list(enumerate(clip_args_list))):
                results.append(res)

        return results
    except Exception as e:
        print(f"Batch LR-ASD pipeline failed, falling back to per-clip: {e}")
        # Fallback to original per-clip generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(clip_args_list), 4)) as executor:
            results = list(executor.map(generate_raw_clip_threadsafe, clip_args_list))
        return results

def generate_raw_clip(base_dir: pathlib.Path,
                               original_video_path: str,
                               s3_key: str,
                               start_time: float,
                               end_time: float,
                      clip_index: int) -> tuple[str, str]:
    """
    Process a single clip: cut, verticalize with LR-ASD, and upload raw clip.

    Returns:
        (raw_clip_url: str, output_s3_key: str)
    """
    clip_name = f"clip_{clip_index}_raw"
    s3_key_dir = os.path.dirname(s3_key)
    # The output key for the raw vertical clip
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Generating raw clip {clip_index} for S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    
    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -y -ss {start_time} -t {duration} -i {original_video_path} "
                   f"-c copy {clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -y -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")

    subprocess.run(columbia_command, cwd="/LR-ASD", shell=True, check=True)

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError(
            f"Tracks or scores not found for clip {clip_index}")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    create_vertical_video(
        tracks, scores, str(pyframes_path), str(
            pyavi_path), str(audio_path), str(vertical_mp4_path)
    )

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        region_name="auto"
    )

    s3_client.upload_file(
        str(vertical_mp4_path), "ai-podcast-clipper", output_s3_key,
        ExtraArgs={'ContentType': 'video/mp4'}
    )

    raw_clip_url = f"https://castclip.revolt-ai.com/{output_s3_key}"
    print(f"Uploaded raw clip {clip_index} to {raw_clip_url}")

    return raw_clip_url, output_s3_key


def generate_raw_clip_threadsafe(args):
    """
    Thread-safe wrapper for generate_raw_clip.
    Generates raw clip only - styling handoff happens later with correct clip numbers.
    """
    clip_id = str(uuid.uuid4())
    
    try:
        # Generate raw vertical clip
        raw_clip_url, output_s3_key = generate_raw_clip(
            base_dir=args["base_dir"],
            original_video_path=args["video_path"],
            s3_key=args["s3_key"],
            start_time=args["start"],
            end_time=args["end"],
            clip_index=args["index"]
        )
        
        return {
            "index": args["index"], 
            "status": "success",
            "clip_id": clip_id,
            "raw_clip_url": raw_clip_url,
            "s3_key": output_s3_key,
            "transcript_segments": args["transcript_segments"],
            "hook": args["hook"],
            "start": args["start"],
            "end": args["end"]
        }
    except Exception as e:
        print(
            f"❌ Raw clip generation for index {args['index']} failed: {str(e)}")
        # Propagate exception details
        import traceback
        traceback.print_exc()
        return {
            "index": args["index"], 
            "status": "error", 
            "error": str(e),
            "clip_id": clip_id
        }


@app.cls(gpu="L40S", timeout=3600, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_modal(self):
        print("Loading models")
        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en",  # maybe change to auto-detect language in future
            device="cuda"
        )

        print("Transcription models loaded...")

        print("Generating Gemini Client")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Generated Gemini Client")

    @modal.method()
    def _process_video_job(self, request_dict: dict, run_id: str):
        """
        Raw clip generation pipeline with parallel styling handoff.
        
        Flow:
        1. Transcribe and align audio with WhisperX
        2. Identify clip-worthy moments via Gemini (with hooks)
        3. For each moment (in parallel):
           - Cut segment, verticalize with LR-ASD, upload raw (clip_{i}_raw.mp4)
           - Immediately call styling endpoint with clip data
        4. Styling endpoint handles final webhook
        5. Only send error webhook if catastrophic failure (all clips failed)
        """
        # Re-hydrate request model (ensures validation) without coupling to FastAPI context
        request = ProcessVideoRequest(**request_dict)
        try:
            s3_key = request.s3_key
            clip_count = request.clip_count

            base_dir = pathlib.Path("/tmp") / run_id
            base_dir.mkdir(parents=True, exist_ok=True)

            # Download Video
            video_path = base_dir / "input.mp4"

            if request.s3_key.startswith("http://") or request.s3_key.startswith("https://"):
                print("Detected YouTube URL, starting download...")
                try:
                    self.download_youtube_video(request.s3_key, video_path)
                    s3_key = f"{request.user_id}/{request.project_id}/input.mp4"
                except HTTPException:
                    # Re-raise the specific YouTube error
                    raise
                except Exception as e:
                    # Catch any other unexpected errors
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error_type": "youtube_download_failed",
                            "message": "Unexpected error during YouTube download",
                            "original_error": str(e)
                        }
                    )
            else:
                print("Downloading video from S3...")
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                    endpoint_url=os.environ["R2_ENDPOINT_URL"],
                    region_name="auto"
                )
                s3_client.download_file(
                    "ai-podcast-clipper", s3_key, str(video_path))

            # 1. Transcription
            transcript_segments_json = self.transcribe_video(
                base_dir, video_path)
            
            transcript_segments = json.loads(transcript_segments_json)

            # 2. Identify moments for clips
            print("Identifying clip moments")
            identified_moments_raw = self.identify_moments(
                transcript_segments, clip_count)

            all_moments = []
            for chunk_str in identified_moments_raw:
                cleaned = chunk_str.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[len("```json"):].strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-len("```")].strip()

                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        all_moments.extend(parsed)
                    else:
                        print("Warning: Parsed chunk is not a list, skipping")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON chunk: {e}")
                    continue

            clip_moments = all_moments

            # 3. Process clips: raw vertical videos with immediate styling handoff
            clip_args_list = []
            for index, moment in enumerate(clip_moments[:clip_count]):
                if "start" in moment and "end" in moment:
                    duration = moment["end"] - moment["start"]
                    if duration >= 10:
                        hook = moment.get("hook", "")
                        
                        # Filter the full transcript to get segments only for this clip
                        # AND adjust timestamps to be relative to clip start (start at 0)
                        clip_start_time = moment["start"]
                        clip_specific_transcript = [
                            {
                                "word": s["word"],
                                "start": s["start"] - clip_start_time,
                                "end": s["end"] - clip_start_time
                            }
                            for s in transcript_segments 
                            if s.get("start", 0) >= moment["start"] and s.get("end", 0) <= moment["end"]
                        ]

                        print(
                            f"Queueing clip {index}: Hook = '{hook}', Duration = {duration:.1f}s")
                        clip_args_list.append({
                            "base_dir": base_dir,
                            "video_path": str(video_path),
                            "s3_key": s3_key,
                            "start": moment["start"],
                            "end": moment["end"],
                            "index": index,
                            "transcript_segments": clip_specific_transcript,
                            "hook": hook,
                            "user_id": request.user_id,
                            "project_id": request.project_id,
                            "caption_style_id": request.style,
                        })
            
            # DON'T set clip_number/total_clips yet - we need to generate first
            # to know which clips will succeed
            
            # Optimized threading with dynamic worker count
            # Limit workers to avoid overwhelming system
            max_workers = min(len(clip_args_list), 4)
            print(
                f"Processing {len(clip_args_list)} raw clips (will determine final count after generation)...")
            
            if clip_args_list:
                # PHASE 1: Generate all raw clips using a single LR-ASD pass
                results = generate_raw_clips_batched(
                    base_dir=base_dir,
                    original_video_path=str(video_path),
                    s3_key=s3_key,
                    clip_args_list=clip_args_list,
                )

                successful_raw_clips = [
                    r for r in results if r["status"] == "success"]
                failed_raw_clips = [r for r in results if r["status"] == "error"]
                
                print(
                    f"🎉 Raw clip generation complete: {len(successful_raw_clips)} successful, {len(failed_raw_clips)} failed")
                
                if failed_raw_clips:
                    for failed_clip in failed_raw_clips:
                        print(
                            f"❌ Failed raw clip {failed_clip['index']}: {failed_clip.get('error', 'Unknown error')}")
                
                # Track styling handoff results and collect payloads
                payloads = []
                if successful_raw_clips:
                    successful_raw_clips.sort(key=lambda x: x['index'])
                    
                    # PHASE 2: Send clips to styling service with correct metadata
                    actual_total_clips = len(successful_raw_clips)
                    print(f"📤 Sending {actual_total_clips} successful clips to styling service...")
                    
                    styling_results = []
                    for idx, clip in enumerate(successful_raw_clips):
                        clip_number = idx + 1
                        is_last = (clip_number == actual_total_clips)
                        
                        # Call styling endpoint with correct metadata
                        styling_success, styling_error, styling_payload = call_styling_endpoint(
                            user_id=request.user_id,
                            project_id=request.project_id,
                            clip_id=clip["clip_id"],
                            s3_key=clip["s3_key"],
                            transcript=clip["transcript_segments"],
                            hook=clip["hook"],
                            caption_style_id=request.style,
                            start_time=clip["start"],
                            end_time=clip["end"],
                            clip_number=clip_number,
                            total_clips=actual_total_clips
                        )
                        
                        styling_results.append(styling_success)
                        print(f"📝 Clip {clip['index']}: clipIndex={clip_number}/{actual_total_clips}, isLastClip={is_last}, styling={'✅' if styling_success else '❌'}")
                    
                    # Check if any clips were successfully sent to styling
                    successful_styling_count = sum(styling_results)
                    
                    if successful_styling_count == 0:
                        # All styling calls failed
                        print(f"❌ All {actual_total_clips} clips failed to send to styling service")
                    elif successful_styling_count < actual_total_clips:
                        # Some styling calls failed
                        print(f"⚠️ {successful_styling_count}/{actual_total_clips} clips sent to styling service successfully")
                    else:
                        # All styling calls succeeded
                        print(f"✅ Job complete. {actual_total_clips} clips sent to styling service.")
                
                # Log if all clips failed to generate raw
                if failed_raw_clips and len(failed_raw_clips) == len(results):
                    # Catastrophic failure - all raw clips failed
                    print("❌ All clips failed to generate raw clips")
                
                return []
            else:
                # No valid moments found
                print("⚠️ No valid moments found")
                return []

        except Exception as e:
            print(f"Unhandled error during process_video: {e}")
            import traceback
            traceback.print_exc()
            return []
            
        finally:
            # Clean up any temp artifacts
            try:
                base_dir = pathlib.Path("/tmp") / run_id
                if base_dir.exists():
                    print(f"Cleaning up temp dir after {base_dir}")
                    shutil.rmtree(base_dir, ignore_errors=True)
            except Exception as cleanup_err:
                logging.error(f"Cleanup error: {cleanup_err}")

    def download_youtube_video(self, youtube_url: str, output_path: pathlib.Path):
        ydl_opts = {
            'format': 'bv*+ba/b',  # More flexible format selection
            'outtmpl': str(output_path),
            'merge_output_format': 'mp4',
            'quiet': False,  # Enable logging to see what's happening
            'noplaylist': True,
            'nocheckcertificate': True,
            'retries': 3,  # Increase retries
            'cookiefile': '/cookies.txt',
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                # First, try to extract info without downloading
                info = ydl.extract_info(youtube_url, download=False)
                
                # Check if video is available
                if not info.get('formats'):
                    raise Exception("No video formats available")
                    
                # Now download
                ydl.download([youtube_url])
                return info.get("title", "Untitled Video")
        except Exception as e:
            print(f"YouTube download error: {str(e)}")
            raise HTTPException(
                status_code=422, 
                detail={
                    "error_type": "youtube_download_failed",
                    "message": "Failed to download YouTube video. This could be due to bot protection, private video, or invalid URL.",
                    "original_error": str(e)
                }
            )

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / 'audio.wav'
        extract_cmd = f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        print("Starting transcription with WhisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print("Transcription and alignment took " + str(duration) + " seconds")

        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"]
                })

        return json.dumps(segments)

    def get_prompt(self, transcript: dict, clip_count: int):
        return f""" 
        You will be given the transcript of a video podcast that will include the start and end times for each word. Podcast owners often times create clips from their podcasts to post on platforms like tiktok, reels and shorts. Your job is to carefully go through the entire podcast and find moments that can are "clip-worthy." Here are some general standards that usually make good clips:

        1. Stories: FInd moments in the podcast where someone is telling a story.
        2. Value/Advice: Sometimes a guest on the podcast will give some sort of advice that can be beneficial to the viewer. 
        3. Question+Answer: This ties into the value/advice category as well as the stories category but I have it seperate because I want you to also prioritize the question that someone asks in the podcast as well as the complete answer (usually value or story), not just the answer.
        4. Written Hook: This is by far the most important part. A hook is supposed to be short (7 words max), it should capture the viewer's attention and keep them curious and engaged throughout the entirety of the clip. You will be focusing on written hooks.
           The hooks you will create will be displayed on the final clips verbatim so be careful. If possible try to incorporate names/identity of the people in the podcast into your hooks, this can insentivise the viewers to want to watch more. Include the name when the person is 
           recognizable and known. If the person isn't recognizable, specify something important about that person (usually the reason they are even on the podcast). For example if the guest is some random person named John and he has made $1 million, the hook shouldn't revolve around John, because no one knows who John is. 
           Instead it should revolve around the million dollars he made. In cases when neither of these cases are met, don't bother to incorporate names/identity of the people in the podcast into your hooks. Also refrain from mentioning the name of anything else that isn't notable, instead try to put a curious spin on it. I
           say this because providing names of services, people, books, or anything that people don't know about can actually put them off and take them away from the video, which will be contridictary to what we are trying to do.

        These are STRICT rules I want you to abide by when creating these clips:

        **Adhere to Timestamps**: Use ONLY the start and end timestamps provided in the input. Do NOT modify the timestamps. The start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
        
        **Ensure Non-Overlapping Clips**: Make sure that the selected clips do not overlap with one another. Each clip should be a distinct segment of the podcast.
        
        **Clip Length**: Aim to generate clips that are between 20 and 120 seconds long. Include as much relevant content as possible within this timeframe.
       
        **Exclude Irrelevant Content**: Exclude the following in your clips:
            - Moments of greeting, thanking, or saying goodbye. This includes introductions that may be good hooks but don't actually provide value at the end. (eg. usually end with "let's get into it")
            - Interactions or segments that do not provide value to the audience.
        
        **Assume No Context**: Assume the person viewing any of the clips you are generating has no prior knowledge of what is happening in the podcast. The clips you create will be their first impression of the podcast and the people in it.

        **Format Output as JSON**: Format your output as a list of JSON objects, with each object representing a clip and containing 'start' and 'end' timestamps in seconds, and a 'hook' to help captivate the viewer as soon as they read it to incentivize them to watch the whole clip. The output should be readable by Python's `json.loads` function.

                        ```json
                        [
                        {{"start": seconds, "end": seconds, "hook": "written hook"}},
                        ...clip2,
                        clip3
                        ]
                        ```
        **Handle No Valid Clips**: If there are no valid clips to extract based on the above criteria, output an empty list in JSON format: `[]`. This output should also be readable by `json.loads()` in Python.

        This is the transcript: {json.dumps(transcript, indent=2)}
        The podcaster has requested you generate {clip_count} clips.
        """

    def split_transcript(self, transcript: dict):
        keys = list(transcript.keys())
        midpoint = len(keys) // 2
        return (
            {k: transcript[k] for k in keys[:midpoint]},
            {k: transcript[k] for k in keys[midpoint:]},
        )

    def split_until_valid(self, transcript: dict, token_limit: int):
        chunks = [transcript]
        final_chunks = []

        while chunks:
            current = chunks.pop(0)
            token_count = self.gemini_client.models.count_tokens(
                model="gemini-2.5-pro",
                contents=str(current)
            )
            if token_count.total_tokens <= token_limit:
                final_chunks.append(current)
            else:
                part1, part2 = self.split_transcript(current)
                chunks.extend([part1, part2])

        return final_chunks

    def identify_moments(self, transcript: dict, clip_count: int):
        # max tokens = 1,048,576
        # prompt + file tokens = 2431
        AVAILABLE_TOKENS = 1046145

        valid_transcripts = self.split_until_valid(
            transcript, AVAILABLE_TOKENS)

        pdf_url = "https://castclip.revolt-ai.com/app/Why%20clips%20went%20viral.pdf"
        pdf_data = httpx.get(pdf_url).content
        max_retries = 2  # number of attempts for primary model before fallback
        clip_count_per_chunk = max(1, clip_count // len(valid_transcripts))
        results = []

        for chunk in valid_transcripts:
            prompt = self.get_prompt(chunk, clip_count_per_chunk)
            primary_model = "gemini-2.5-pro"
            fallback_model = "gemini-2.5-flash"

            success = False

            # Try primary model with retries and 5s backoff
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=primary_model,
                        config=types.GenerateContentConfig(safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                        ],
                            system_instruction="You are a Podcast Clip Extractor, tasked with creating clips for short-form platforms like TikTok and YouTube Shorts. Your goal is to identify and extract engaging stories, questions, and answers from podcast transcripts that will appeal to up-and-coming entrepreneurs seeking advice and motivation."),
                        contents=[
                            types.Content(role="user", parts=[types.Part(text=prompt)]),
                            types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")
                        ]
                    )

                    if response.text:
                        results.append(response.text)
                        success = True
                        break
                    else:
                        print(f"Attempt {attempt} ({primary_model}): Empty response.text from Gemini")
                except Exception as e:
                    print(f"Attempt {attempt} ({primary_model}) failed - {e}")

                if attempt < max_retries:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)

            # Fallback to flash model if primary failed
            if not success:
                try:
                    print(f"Falling back to {fallback_model}...")
                    response = self.gemini_client.models.generate_content(
                        model=fallback_model,
                        config=types.GenerateContentConfig(safety_settings=[
                            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                        ],
                            system_instruction="You are a Podcast Clip Extractor, tasked with creating clips for short-form platforms like TikTok and YouTube Shorts. Your goal is to identify and extract engaging stories, questions, and answers from podcast transcripts that will appeal to up-and-coming entrepreneurs seeking advice and motivation."),
                        contents=[
                            types.Content(role="user", parts=[types.Part(text=prompt)]),
                            types.Part.from_bytes(data=pdf_data, mime_type="application/pdf")
                        ]
                    )
                    if response.text:
                        results.append(response.text)
                    else:
                        print(f"Fallback ({fallback_model}) produced empty response.text")
                except Exception as e:
                    print(f"Fallback ({fallback_model}) failed - {e}")

        return results if results else "[]"

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        # Authenticate
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect Bearer Token", headers={"WWW-Authenticate": "Bearer"})

        # Generate a job id
        run_id = str(uuid.uuid4())
        
        # Immediately return 202 Accepted
        # Processing will happen asynchronously and styling service will notify frontend
        print(f"✅ Accepted job {run_id} - starting processing...")
        
        # Spawn the job in the background (Modal handles this)
        self._process_video_job.spawn(request.model_dump(), run_id)

        # Return 202 Accepted immediately
        return JSONResponse(status_code=202, content={
            "job_id": run_id,
            "status": "accepted",
            "message": "Video processing job accepted and started."
        })


@app.local_entrypoint()
def main():
    """
    Local test entrypoint for the raw clip generation + styling handoff pipeline.
    Tests the process_video endpoint with a sample payload.
    Writes styling payloads to local files for debugging (payload_clip_0.txt, etc.)
    """
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    url = ai_podcast_clipper.process_video.web_url

    payload = {
        "s3_key": "https://youtu.be/wm-hqM9F8BM?si=OYLAHbJ8oCX9t0-R",
        "user_id": "3zhHBEDMtMIEiRshOHkFSV72wKfdKkKI",
        "project_id": "123",
        "clip_count": 3,
        "style": 1
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer DhfvXdZywkp8PTdAYjQLrBbgBbdSztfiqZPGXdeyY96msKe6gmJXZz93fGkGUsmh"
    }

    print("Sending request to process_video endpoint...")
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    result = response.json()
    print(f"Response: {result}")
    
    # Write payloads to local files
    if result.get("payloads"):
        print(f"\n📝 Writing {len(result['payloads'])} payload(s) to local files...")
        for payload_data in result["payloads"]:
            clip_index = payload_data["clip_index"]
            clip_payload = payload_data["payload"]
            
            filename = f"payload_clip_{clip_index}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(clip_payload, f, indent=2, ensure_ascii=False)
            print(f"✅ Written {filename}")
        print(f"\n🎉 All payloads written successfully!")
    else:
        print("\n⚠️ No payloads returned from endpoint")
