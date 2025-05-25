import modal
from fastapi import Depends, status, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, validator, Field
from typing import Optional, Union
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
import pysubs2
from google.genai import types
import yt_dlp
import re
import logging

class ProcessVideoRequest(BaseModel):
    s3_key: Optional[str] = None
    youtube_url: Optional[str] = None
    clip_count: int = Field(default=1, ge=1, le=10)
    
    @validator('youtube_url')
    def validate_youtube_url(cls, v):
        if v is not None:
            # Regex pattern for YouTube URLs
            youtube_pattern = re.compile(
                r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
                r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
            )
            if not youtube_pattern.match(v):
                raise ValueError('Invalid YouTube URL format')
        return v
    
    @validator('s3_key', always=True)
    def validate_input_source(cls, v, values):
        s3_key = v
        youtube_url = values.get('youtube_url')
        
        if not s3_key and not youtube_url:
            raise ValueError('Either s3_key or youtube_url must be provided')
        if s3_key and youtube_url:
            raise ValueError('Provide either s3_key or youtube_url, not both')
        
        return v

image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("LR-ASD", "/LR-ASD", copy=True))

app = modal.App("ai-podcast-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clipper-modal-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()

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
                    {'track': tidx, 'score': avg_score, 's': track['proc_track']["s"][fidx], 'x': track['proc_track']["x"][fidx], 'y': track['proc_track']["y"][fidx]}
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

    ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)

def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str, max_words: int = 5):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]

    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= 0:
            continue

        if not current_words:
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        elif len(current_words) >= max_words:
            subtitles.append(
                (current_start, current_end, ' '.join(current_words)))
            current_words = [word]
            current_start = start_rel
            current_end = end_rel
        else:
            current_words.append(word)
            current_end = end_rel

    if current_words:
        subtitles.append(
            (current_start, current_end, ' '.join(current_words)))

    subs = pysubs2.SSAFile()

    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 140
    new_style.primarycolor = pysubs2.Color(255, 255, 255)
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 5
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 50
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(
            start=start_time, end=end_time, text=text, style=style_name)
        subs.events.append(line)

    subs.save(subtitle_path)

    ffmpeg_cmd = (f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                  f"-c:v h264 -preset fast -crf 23 {output_path}")

    subprocess.run(ffmpeg_cmd, shell=True, check=True)
  
def process_clip(base_dir: str, original_video_path: str, output_s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
    clip_name = f"clip_{clip_index}"
    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")

    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/LR-ASD", shell=True)
    columbia_end_time = time.time()
    print(
        f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(
        f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")
    
    create_subtitles_with_ffmpeg(transcript_segments, start_time, end_time, vertical_mp4_path, subtitle_output_path, max_words=5)
    
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        region_name="auto"  # Cloudflare R2 ignores region, but boto3 still requires it
    )

    s3_client.upload_file(subtitle_output_path, "ai-podcast-clipper", output_s3_key)

@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_modal(self):
        print("Loading models")
        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")
        
        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en", # maybe change to auto-detect language in future
            device="cuda"
        )

        print("Transcription models loaded...")

        print("Generating Gemini Client")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Generated Gemini Client")

    def download_youtube_video(self, youtube_url: str, output_path: str) -> tuple[str, dict]:
        """
        Download a YouTube video using yt-dlp.
        
        Args:
            youtube_url: The YouTube URL to download
            output_path: Path where to save the video
            
        Returns:
            tuple: (downloaded_file_path, video_info)
        """
        print(f"Downloading YouTube video: {youtube_url}")
        
        # Configure yt-dlp options for high quality video
        ydl_opts = {
            'format': 'best[height<=1080][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'writeinfojson': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'ignoreerrors': False,
            'no_warnings': False,
            'extractflat': False,
            'writethumbnail': False,
            'noplaylist': True,  # Only download the specific video, not playlist
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info first
                info = ydl.extract_info(youtube_url, download=False)
                
                # Check duration (limit to reasonable podcast length)
                duration = info.get('duration', 0)
                if duration > 10800:  # 3 hours limit
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Video too long ({duration/3600:.1f} hours). Maximum allowed is 3 hours."
                    )
                
                if duration < 60:  # 1 minute minimum
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Video too short ({duration} seconds). Minimum required is 1 minute."
                    )
                
                print(f"Video info: Title='{info.get('title', 'Unknown')}', Duration={duration}s")
                
                # Download the video
                ydl.download([youtube_url])
                
                return output_path, info
                
        except yt_dlp.DownloadError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download YouTube video: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error downloading video: {str(e)}"
            )

    def download_s3_video(self, s3_key: str, output_path: str) -> str:
        """
        Download a video from S3.
        
        Args:
            s3_key: The S3 key for the video
            output_path: Path where to save the video
            
        Returns:
            str: The path to the downloaded video
        """
        print(f"Downloading video from S3: {s3_key}")
        
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            region_name="auto"
        )
        
        try:
            s3_client.download_file("ai-podcast-clipper", s3_key, output_path)
            return output_path
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download video from S3: {str(e)}"
            )

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / 'audio.wav'
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

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
        print("Transcription and alignment took " + str(duration) +  " seconds")

        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"]
                })
        
        return json.dumps(segments)
    
    def identify_moments(self, transcript: dict, clip_count: int):
        print(str(transcript))

        response = self.gemini_client.models.generate_content(model="gemini-2.5-flash-preview-05-20", 
                                                              config=types.GenerateContentConfig(safety_settings=[
                                                                types.SafetySetting( category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"), 
                                                                types.SafetySetting( category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE" ), 
                                                                types.SafetySetting( category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE" ), 
                                                                types.SafetySetting( category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"), 
                                                              ]),
                                                              contents=f"""
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 20 and maximum of 60 seconds long. The clip should never exceed 60 seconds or be under 20 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer or be the full story that is being told.
    It is acceptable for the clip to include a few additional sentences before a question/story if it aids in contextualizing the question/story.

    Please adhere to the following rules:
     - Ensure that clips do not overlap with one another.
     - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
     - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
     - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{{"start": seconds, "end": seconds}}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
     - Generate exactly {clip_count} clips. This is the exact number requested by the user.
     - Aim to generate longer clips between 20-60 seconds, and ensure to include as much content from the context as viable.
    
    Avoid including:
     - Moments of greeting, thanking, or saying goodbye.
     - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n""" + str(transcript))
        print(response.model_dump_json())
        return response.text

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key
        youtube_url = request.youtube_url
        clip_count = request.clip_count

        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect Bearer Token", headers={"WWW-Authenticate": "Bearer"})
        
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        video_info = None
        video_source = None
        
        try:
            # Download Video from appropriate source
            video_path = base_dir / "input.mp4"
            
            if youtube_url:
                video_source = "youtube"
                print(f"Processing YouTube video: {youtube_url}")
                video_path_str, video_info = self.download_youtube_video(youtube_url, str(video_path))
                
                # Generate S3 key for output based on YouTube video ID
                video_id = video_info.get('id', run_id)
                video_title = video_info.get('title', 'unknown')
                # Sanitize title for use in S3 key
                sanitized_title = re.sub(r'[^\w\-_\.]', '_', video_title)[:50]
                output_s3_prefix = f"youtube/{video_id}/{sanitized_title}"
                
            elif s3_key:
                video_source = "s3"
                print(f"Processing S3 video: {s3_key}")
                self.download_s3_video(s3_key, str(video_path))
                output_s3_prefix = os.path.dirname(s3_key)
            
            print(f"Video downloaded successfully to: {video_path}")
            
            # 1. Transcription
            print("Starting transcription...")
            transcript_segments_json = self.transcribe_video(base_dir, video_path)
            transcript_segments = json.loads(transcript_segments_json)

            # 2. Identify moments for clips
            print("Identifying clip moments...")
            identified_moments_raw = self.identify_moments(transcript_segments, clip_count)

            cleaned_json_string = identified_moments_raw.strip()
            if cleaned_json_string.startswith("```json"):
                cleaned_json_string = cleaned_json_string[len("```json"):].strip()
            if cleaned_json_string.endswith("```"):
                cleaned_json_string = cleaned_json_string[:-len("```")].strip()

            clip_moments = json.loads(cleaned_json_string)
            if not isinstance(clip_moments, list):
                print("Error: Identified moments is not a list")
                clip_moments = []
            
            print(f"Found {len(clip_moments)} potential clips")

            # 3. Process Clips
            processed_clips = []
            for index, moment in enumerate(clip_moments[:clip_count]):
                if "start" in moment and "end" in moment:
                    duration = moment["end"] - moment["start"]
                    if duration >= 10: 
                        print(f"Processing clip {index} from {moment['start']} to {moment['end']} ({duration:.1f}s)")
                        
                        # Generate output S3 key for this clip
                        clip_s3_key = f"{output_s3_prefix}/clip_{index}.mp4"
                        
                        try:
                            process_clip(base_dir, video_path, clip_s3_key, moment["start"], moment["end"], index, transcript_segments)
                            processed_clips.append({
                                "clip_index": index,
                                "start_time": moment["start"],
                                "end_time": moment["end"],
                                "duration": duration,
                                "s3_key": clip_s3_key,
                                "status": "success"
                            })
                        except Exception as e:
                            print(f"Error processing clip {index}: {str(e)}")
                            processed_clips.append({
                                "clip_index": index,
                                "start_time": moment["start"],
                                "end_time": moment["end"],
                                "duration": duration,
                                "s3_key": clip_s3_key,
                                "status": "error",
                                "error": str(e)
                            })
                    else:
                        print(f"Skipping clip {index}: too short ({duration:.1f}s)")

            # Prepare response
            response_data = {
                "run_id": run_id,
                "video_source": video_source,
                "processed_clips": processed_clips,
                "total_clips_processed": len([c for c in processed_clips if c["status"] == "success"]),
                "total_clips_requested": clip_count
            }
            
            # Add source-specific metadata
            if video_source == "youtube":
                response_data["video_info"] = {
                    "title": video_info.get("title"),
                    "duration": video_info.get("duration"),
                    "uploader": video_info.get("uploader"),
                    "upload_date": video_info.get("upload_date"),
                    "view_count": video_info.get("view_count")
                }
            elif video_source == "s3":
                response_data["s3_key"] = s3_key

            return response_data

        except Exception as e:
            print(f"Error in process_video: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing video: {str(e)}"
            )
        finally:
            # Clean up temporary files
            if base_dir.exists():
                print(f"Cleaning up temp dir: {base_dir}")
                shutil.rmtree(base_dir, ignore_errors=True)

@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()
    url = ai_podcast_clipper.process_video.web_url

    # Example 1: Process from S3 (original functionality)
    print("=== Example 1: Processing from S3 ===")
    s3_payload = {
        "s3_key": "3zhHBEDMtMIEiRshOHkFSV72wKfdKkKI/3da8790a-195b-4314-b940-bb5d8819f462/original.mp4",
        "clip_count": 1
    }
    
    # Example 2: Process from YouTube URL
    print("=== Example 2: Processing from YouTube ===")
    youtube_payload = {
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Replace with actual podcast URL
        "clip_count": 2
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    # Choose which example to run (comment/uncomment as needed)
    payload = youtube_payload  # Change to s3_payload to test S3 functionality
    
    print(f"Sending request to: {url}")
    print(f"Payload: {payload}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=900)
        response.raise_for_status()
        result = response.json()
        
        print("=== Processing completed successfully ===")
        print(f"Run ID: {result.get('run_id')}")
        print(f"Video Source: {result.get('video_source')}")
        print(f"Clips processed: {result.get('total_clips_processed')}/{result.get('total_clips_requested')}")
        
        if result.get('video_source') == 'youtube':
            video_info = result.get('video_info', {})
            print(f"Video Title: {video_info.get('title')}")
            print(f"Duration: {video_info.get('duration')}s")
            print(f"Uploader: {video_info.get('uploader')}")
        
        print("\n=== Processed Clips ===")
        for clip in result.get('processed_clips', []):
            if clip['status'] == 'success':
                print(f"✓ Clip {clip['clip_index']}: {clip['start_time']:.1f}s - {clip['end_time']:.1f}s ({clip['duration']:.1f}s)")
                print(f"  S3 Key: {clip['s3_key']}")
            else:
                print(f"✗ Clip {clip['clip_index']}: Failed - {clip.get('error', 'Unknown error')}")
                
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if e.response.content:
            print(f"Error details: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
