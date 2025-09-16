import modal
from fastapi import Depends, status, HTTPException, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ConfigDict
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
from yt_dlp import YoutubeDL
import httpx
import concurrent.futures
import requests
import logging
from typing import Optional, List
from fastapi.responses import JSONResponse
from urllib.parse import urlparse


def send_completion_webhook(webhook_url: str, user_id: str, project_id: str, status: str = "completed", error_message: Optional[str] = None, clips: Optional[list] = None) -> bool:
    """
    Sends a secure POST request to the provided webhook URL to notify
    that video processing is complete or failed.

    Args:
        webhook_url: The callback URL provided by the Next.js app
        user_id: The ID of the user
        project_id: The ID of the project that was processed
        status: Processing status ('completed', 'failed', 'error', 'ready_for_review')
        error_message: Optional error message if status is 'failed' or 'error'
        clips: Optional list of clip data for the 'ready_for_review' status
    
    Returns:
        bool: True if webhook was sent successfully, False otherwise
    """
    if not webhook_url or not webhook_url.strip():
        logging.warning("Webhook URL not provided, skipping notification")
        return False

    print(f"Webhook URL: {webhook_url}")
    
    secret_token = os.getenv("WEBHOOK_SECRET")
    if not secret_token:
        logging.critical("WEBHOOK_SECRET environment variable not set")
        return False

    # Parse user_id and project_id from ids if needed
    if "/" in user_id:
        user_id, project_id = user_id.split("/", 1)

    # Construct payload with comprehensive information
    payload = {
        "user_id": user_id,
        "project_id": project_id,
        "status": status,
        "timestamp": int(time.time()),
        "processing_completed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    
    # Add error information if applicable
    if error_message and status in ["failed", "error"]:
        payload["error"] = {
            "message": error_message,
            "type": "processing_error"
        }
    
    # Add clips data if the analysis stage is complete
    if clips and status == "ready_for_review":
        payload["clips"] = clips

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {secret_token}",
        "User-Agent": "AI-Podcast-Clipper/1.0",
        "X-Webhook-Source": "modal-backend"
    }

    logging.info(
        f"Sending {status} webhook for project {project_id} to {webhook_url}")

    try:
        # Use a reasonable timeout and retry configuration
        response = requests.post(
            webhook_url, 
            json=payload, 
            headers=headers, 
            timeout=30,
            allow_redirects=False  # Don't follow redirects for security
        )

        # Log response details for debugging
        logging.info(
            f"Webhook response: {response.status_code} for project {project_id}")
        
        # Check for successful response (2xx status codes)
        if 200 <= response.status_code < 300:
            logging.info(
                f"Successfully sent {status} webhook for project {project_id}")
            return True
        else:
            logging.error(
                f"Webhook failed with status {response.status_code} for project {project_id}")
            # Log first 500 chars
            logging.error(f"Response body: {response.text[:500]}")
            return False

    except requests.exceptions.Timeout:
        logging.error(
            f"Webhook timeout for project {project_id} after 30 seconds")
        return False
    except requests.exceptions.ConnectionError:
        logging.error(
            f"Connection error sending webhook for project {project_id}")
        return False
    except requests.exceptions.RequestException as e:
        logging.error(
            f"Failed to send webhook for project {project_id}: {type(e).__name__}: {str(e)}")
        return False
    except Exception as e:
        logging.error(
            f"Unexpected error sending webhook for project {project_id}: {type(e).__name__}: {str(e)}")
        return False


class ProcessVideoRequest(BaseModel):
    s3_key: str
    ids: str          # userid/projectid
    clip_count: int = 1
    style: int = 1
    webhook_url: str  # Optional webhook URL for completion notifications


# ---- NEW: Pydantic model for the final rendering request ----
# This defines the data structure the frontend will send when a user
# wants to export a clip with their final edits.
class Position(BaseModel):
    x: float
    y: float


class StyleOptions(BaseModel):
    fontSize: float
    position: Position


class WordSegment(BaseModel):
    word: str
    start: float
    end: float


class RenderVideoRequest(BaseModel):
    clip_id: str
    raw_clip_url: str = Field(..., alias='raw_clip_url')
    transcript_segments: List[WordSegment] = Field(
        ..., alias='transcript_segments')
    hook: Optional[str] = None
    hook_style: Optional[StyleOptions] = Field(None, alias='hook_style')
    project_style: int = Field(1, alias='project_style')
    captions_style: Optional[StyleOptions] = Field(
        None, alias='captions_style')
    # New field to specify subtitle position, e.g., {"x": 540, "y": 1600}
    caption_position: Optional[dict] = None

    model_config = ConfigDict(extra='ignore')


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "wget -O /usr/share/fonts/truetype/custom/impact.ttf https://github.com/sophilabs/macgifer/raw/master/static/font/impact.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("LR-ASD", "/LR-ASD", copy=True)
    .add_local_file("emoji_config.json", "/emoji_config.json")
    .add_local_file("emoji_filenames.json", "/emoji_filenames.json")
    .add_local_file("cookies.txt", "/cookies.txt"))

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
                blurred_background, (121, 121), 0)

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

    ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)


class ProductionEmojiManager:
    """
    Production-level emoji manager using R2 CDN
    - Zero local storage
    - Direct URL generation
    - Fast word-to-filename mapping
    - Optimized for Modal deployment
    """
    
    def __init__(self, 
                 emoji_config_path="emoji_config.json",
                 emoji_filenames_path="emoji_filenames.json",
                 r2_base_url="https://castclip.revolt-ai.com/app/emojis"):
        
        self.r2_base_url = r2_base_url.rstrip('/')
        self.word_to_filename = {}
        self.available_filenames = set()
        
        # Load configurations
        self._load_configurations(emoji_config_path, emoji_filenames_path)
        print(
            f"🚀 Production Emoji Manager: {len(self.word_to_filename)} mappings ready")
    
    def _load_configurations(self, emoji_config_path, emoji_filenames_path):
        """Load and cross-reference emoji configs with available filenames"""
        
        # Load available filenames from R2
        try:
            with open(emoji_filenames_path, 'r', encoding='utf-8') as f:
                filenames_list = json.load(f)
                self.available_filenames = set(filenames_list)
                print(
                    f"✅ Loaded {len(self.available_filenames)} available emoji files")
        except FileNotFoundError:
            print(
                f"⚠️ {emoji_filenames_path} not found. Using emoji config only.")
            self.available_filenames = set()
        
        # Load word-to-emoji mappings
        try:
            with open(emoji_config_path, 'r', encoding='utf-8') as f:
                emoji_data = json.load(f)
                self._create_word_to_filename_mapping(emoji_data)
        except FileNotFoundError:
            print(
                f"⚠️ {emoji_config_path} not found. Using minimal fallback mapping.")
            self._create_fallback_mapping()
    
    def _create_word_to_filename_mapping(self, emoji_data):
        """Create efficient word -> filename mapping"""
        
        # Create emoji character to filename mapping
        emoji_to_filename = self._create_emoji_to_filename_map()
        
        mapped_count = 0
        unmapped_count = 0
        
        for emoji_char, words in emoji_data.items():
            filename = emoji_to_filename.get(emoji_char)
            
            if filename and filename in self.available_filenames:
                # Map all words to this filename
                for word in words:
                    word_clean = word.lower().strip()
                    if word_clean and word_clean not in self.word_to_filename:
                        self.word_to_filename[word_clean] = filename
                        mapped_count += 1
            else:
                unmapped_count += len(words)
        
        print(
            f"📊 Mapping Stats: {mapped_count} words mapped, {unmapped_count} words unmapped")
    
    def _create_emoji_to_filename_map(self):
        """Create mapping from emoji characters to filenames"""
        
        # Standard emoji Unicode name mapping
        emoji_name_map = {
            "🔥": "fire.png",
            "❤️": "red_heart.png", 
            "😂": "face_with_tears_of_joy.png",
            "💯": "hundred_points.png",
            "🚀": "rocket.png",
            "💪": "flexed_biceps.png",
            "🎉": "party_popper.png",
            "💰": "money_bag.png",
            "🌟": "glowing_star.png",
            "👑": "crown.png",
            "🎯": "bullseye.png",
            "🧠": "brain.png",
            "👀": "eyes.png",
            "👍": "thumbs_up.png",
            "🙏": "folded_hands.png",
            "😎": "smiling_face_with_sunglasses.png",
            "😢": "crying_face.png",
            "😡": "pouting_face.png",
            "🤯": "exploding_head.png",
            "☀️": "sun.png",
            "🌙": "crescent_moon.png",
            "🌈": "rainbow.png",
            "⏰": "alarm_clock.png",
            "💎": "gem_stone.png",
            "🔑": "key.png",
            "🎤": "microphone.png",
            "💻": "laptop.png",
            "📚": "books.png",
            "🏠": "house.png",
            "✈️": "airplane.png",
            "🎵": "musical_note.png",
            "🍕": "pizza.png",
            "☕": "hot_beverage.png",
            "🍎": "red_apple.png",
            "🎮": "video_game.png",
            "🏃": "person_running.png",
            "🛏️": "bed.png",
            "🚗": "automobile.png",
            "📈": "chart_increasing.png",
            "📉": "chart_decreasing.png",
            "🔒": "locked.png",
            "🌍": "globe_showing_europe-africa.png",
            "🎨": "artist_palette.png",
            "🤝": "handshake.png",
            "🎪": "circus_tent.png",
            "🌺": "hibiscus.png",
            "🍰": "birthday_cake.png",
            "🎁": "wrapped_gift.png",
            "🏖️": "beach_with_umbrella.png",
            "🔊": "speaker_high_volume.png",
            "💭": "thought_balloon.png",
            "📸": "camera_with_flash.png",
            "🛠️": "hammer_and_wrench.png",
            "🎲": "game_die.png",
            "🤷": "person_shrugging.png",
            "🙄": "face_with_rolling_eyes.png",
            "🥺": "pleading_face.png",
            "💅": "nail_polish.png",
            "🧘": "person_in_lotus_position.png",
            "🛒": "shopping_cart.png",
            "🚨": "police_car_light.png"
        }
        
        return emoji_name_map
    
    def _create_fallback_mapping(self):
        """Create minimal fallback mapping if config files missing"""
        fallback_mappings = {
            "fire": "fire.png",
            "sun": "sun.png", 
            "good": "thumbs_up.png",
            "money": "money_bag.png",
            "party": "party_popper.png",
            "love": "red_heart.png",
            "laugh": "face_with_tears_of_joy.png"
        }
        
        for word, filename in fallback_mappings.items():
            if filename in self.available_filenames:
                self.word_to_filename[word] = filename
    
    def get_emoji_url(self, word):
        """
        Get R2 CDN URL for emoji by word
        Returns None if no emoji found
        """
        word_clean = word.lower().strip()
        filename = self.word_to_filename.get(word_clean)
        
        if filename:
            return f"{self.r2_base_url}/{filename}"
        
        return None
    
    def get_stats(self):
        """Get emoji manager statistics for monitoring"""
        return {
            "total_mappings": len(self.word_to_filename),
            "available_files": len(self.available_filenames),
            "coverage_ratio": len(self.word_to_filename) / max(1, len(self.available_filenames)),
            "r2_base_url": self.r2_base_url
        }


def generate_emoji_overlays_production(clip_segments, clip_start, emoji_manager, group_size=3):
    """
    Production emoji overlay generation using R2 URLs
    No local files - returns URLs for direct FFmpeg usage
    
    Args:
        clip_segments: List of transcript segments with word timing
        clip_start: Start time of the clip
        emoji_manager: ProductionEmojiManager instance
        group_size: Number of words per group (default: 3)
    
    Returns:
        List of emoji overlay dictionaries with url, word, start, end
    """
    emoji_overlays = []
    
    i = 0
    while i < len(clip_segments):
        group = clip_segments[i:i + group_size]
        if len(group) < group_size:
            break
            
        # Find emoji word in group
        emoji_url = None
        emoji_word = None
        
        for seg in group:
            word = seg.get("word", "").lower().strip()
            url = emoji_manager.get_emoji_url(word)
            if url:
                emoji_url = url
                emoji_word = word
                break
        
        if emoji_url:
            group_start = group[0]["start"]
            group_end = group[-1]["end"] if i + group_size >= len(clip_segments) else min(
                group[-1]["end"], clip_segments[i + group_size]["start"])
            
            emoji_overlays.append({
                "url": emoji_url,
                "word": emoji_word,
                "start": max(0, group_start - clip_start),
                "end": max(0, group_end - clip_start)
            })
        
        i += group_size
    
    return emoji_overlays


def create_optimized_ffmpeg_command_production(emoji_overlays, clip_video_path, subtitle_path, output_path):
    """
    Production FFmpeg command using R2 URLs
    Maximum efficiency for Modal deployment
    
    Args:
        emoji_overlays: List of emoji overlays from generate_emoji_overlays_production
        clip_video_path: Input video file path
        subtitle_path: ASS subtitle file path  
        output_path: Output video file path
    
    Returns:
        Complete FFmpeg command string ready for execution
    """
    
    if not emoji_overlays:
        # No emojis: simple subtitle overlay
        return f'ffmpeg -y -i "{clip_video_path}" -vf "ass={subtitle_path}" -c:v h264 -preset fast -crf 23 -c:a copy "{output_path}"'
    
    # Build inputs (video + all emoji URLs)
    inputs = [f'-i "{clip_video_path}"']
    
    # Add each unique emoji URL as input
    unique_urls = {}
    for overlay in emoji_overlays:
        url = overlay["url"]
        if url not in unique_urls:
            unique_urls[url] = len(inputs)
            inputs.append(f'-i "{url}"')
    
    # Build filter chain
    filters = []
    
    # Scale each unique emoji input
    unique_scaled_streams = {}
    for url, input_idx in unique_urls.items():
        scaled_stream_name = f"[scaled_emoji_{input_idx}]"
        filters.append(f"[{input_idx}:v]scale=150:150{scaled_stream_name}")
        unique_scaled_streams[url] = scaled_stream_name

    current_output = "[0:v]"
    
    # Group overlays by URL for efficiency
    url_overlays_grouped = {}
    for overlay in emoji_overlays:
        url = overlay["url"]
        if url not in url_overlays_grouped:
            url_overlays_grouped[url] = []
        url_overlays_grouped[url].append(overlay)
    
    filter_idx = 1
    # Apply overlays one by one, creating a chain
    for url, overlays in url_overlays_grouped.items():
        scaled_emoji_stream = unique_scaled_streams[url]
        
        # Create time conditions for this emoji
        time_conditions = []
        for overlay in overlays:
            time_conditions.append(
                f"between(t,{overlay['start']},{overlay['end']})")
        
        enable_condition = "+".join(time_conditions)
        next_output = f"[v{filter_idx}]"
        
        # Position adjusted to be above text and centered
        filters.append(
            f"{current_output}{scaled_emoji_stream} overlay=(main_w-overlay_w)/2:1200:enable='{enable_condition}' {next_output}")
        current_output = next_output
        filter_idx += 1
    
    # Add subtitles last
    filters.append(f"{current_output} ass={subtitle_path} [final]")
    
    filter_chain = "; ".join(filters)
    return f'ffmpeg -y {" ".join(inputs)} -filter_complex "{filter_chain}" -map "[final]" -map 0:a? -c:v h264 -preset fast -crf 23 "{output_path}"'


def create_subtitles_with_ffmpeg(transcript_segments, clip_start, clip_end, clip_video_path, output_path, hook, 
                                style=1, emoji_config_path="/emoji_config.json", 
                                emoji_filenames_path="/emoji_filenames.json",
                                 hook_style_options: Optional[dict] = None,
                                 captions_style_options: Optional[dict] = None):
    
    if style not in [1, 2, 3, 4]:
        raise ValueError("Style must be between 1 and 4")
    
    video_width = 1080
    video_height = 1920
    SCALE_CORRECTION = 1.35
    
    temp_dir = os.path.dirname(output_path) or "."
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    # Get video duration if clip_end not specified
    if clip_end is None:
        import ffmpeg
        clip_end = float(ffmpeg.probe(clip_video_path)["format"]["duration"])

    # Filter segments for the clip timeframe
    clip_segments = [s for s in transcript_segments 
                    if s.get("start") and s.get("end") and 
                    s["end"] > clip_start and s["start"] < clip_end]

    # Create ASS subtitle file
    subs = pysubs2.SSAFile()
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    # ---- NEW: Handle custom caption positioning ----
    position_tag = ""
    if captions_style_options and 'position' in captions_style_options:
        pos_x = int(
            (captions_style_options['position']['x'] / 100) * video_width)
        pos_y = int(
            (captions_style_options['position']['y'] / 100) * video_height)
        position_tag = f"\\pos({pos_x}, {pos_y})"

    # ---- NEW: Handle custom hook positioning ----
    hook_position_tag = ""
    if hook_style_options and 'position' in hook_style_options:
        pos_x = int((hook_style_options['position']['x'] / 100) * video_width)
        pos_y = int((hook_style_options['position']['y'] / 100) * video_height)
        hook_position_tag = f"\\pos({pos_x}, {pos_y})"

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    
    # Configure style based on parameter
    if style == 1:
        # Style 1: Regular subtitles (Anton font, no emojis)
        new_style.fontname = "Anton"
        new_style.fontsize = 140
        new_style.primarycolor = pysubs2.Color(255, 255, 255)
        new_style.outline = 4.0
        new_style.shadow = 2.0
        new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
        new_style.alignment = 2
        new_style.marginl = 50
        new_style.marginr = 50
        new_style.marginv = 600
        new_style.spacing = 0.0
        
    elif style == 2:
        # Style 2: Impact font with emojis and subtle glow effect
        new_style.fontname = "impact"
        new_style.fontsize = 100
        new_style.primarycolor = pysubs2.Color(255, 255, 255)
        new_style.outline = 2.0
        new_style.shadow = 0.0
        new_style.outlinecolor = pysubs2.Color(100, 100, 100)
        new_style.blur = 1.0
        new_style.alignment = 2
        new_style.marginl = 40
        new_style.marginr = 40
        new_style.marginv = 400  # Lowered from 800 to move text down
        new_style.spacing = 2.0
        if captions_style_options and 'fontSize' in captions_style_options:
            new_style.fontsize = captions_style_options['fontSize']
        
    elif style == 3:
        # Style 3: Orange karaoke style (Impact Font)
        new_style.fontname = "impact"  
        new_style.fontsize = 100  
        new_style.primarycolor = pysubs2.Color(255, 255, 255)  
        new_style.secondarycolor = pysubs2.Color(255, 165, 0) 
        new_style.outline = 3.0  
        new_style.outlinecolor = pysubs2.Color(0, 0, 0)  
        new_style.shadow = 2.0  
        new_style.shadowcolor = pysubs2.Color(0, 0, 0, 150) 
        new_style.alignment = 2  
        new_style.marginl = 140  
        new_style.marginr = 140
        new_style.marginv = 500  
        new_style.spacing = 0.5  
        new_style.bold = True  
        new_style.italic = False
        
    elif style == 4:
        # Style 4: TODO - Future implementation  
        raise NotImplementedError("Style 4 not implemented yet")
    
    # Override font size with scaled value from frontend if provided
    if captions_style_options and 'fontSize' in captions_style_options:
        scaled_font_size = int(
            captions_style_options['fontSize'] * (subs.info["PlayResY"] / 600)) * SCALE_CORRECTION
        new_style.fontsize = scaled_font_size
    
    subs.styles[style_name] = new_style

    # Add hook style
    if hook:
        hook_style = pysubs2.SSAStyle()
        hook_style.fontname = "Impact"
        hook_style.fontsize = 100
        hook_style.primarycolor = pysubs2.Color(255, 255, 255)  # White text
        hook_style.outline = 3.0
        hook_style.outlinecolor = pysubs2.Color(0, 0, 0)  # Black outline
        hook_style.shadow = 2.0
        hook_style.shadowcolor = pysubs2.Color(0, 0, 0, 180)  # Dark shadow
        hook_style.alignment = 8  # Center-top alignment (not center-bottom)
        hook_style.marginl = 60
        hook_style.marginr = 60
        hook_style.marginv = 250  # Distance from top edge
        hook_style.bold = True
        hook_style.spacing = 1.0
        
        if hook_style_options:
            if 'fontSize' in hook_style_options:
                hook_style.fontsize = int(
                    hook_style_options['fontSize'] * (subs.info["PlayResY"] / 600)) * SCALE_CORRECTION
            if 'position' in hook_style_options:
                # If a specific position is given, we override alignment and margins
                hook_style.alignment = 5  # Middle-center alignment for pos() tag
                hook_style.marginl, hook_style.marginr, hook_style.marginv = 0, 0, 0
        
        subs.styles["Hook"] = hook_style
        
        # Add title event that spans the entire video
        hook_text = hook.upper()

        subs.events.append(pysubs2.SSAEvent(
            start=pysubs2.make_time(s=0),
            end=pysubs2.make_time(s=clip_end - clip_start if clip_end else 30),
            text=f"{{{hook_position_tag}}}{hook_text}",
            style="Hook"
        ))

    if style == 3:
        group_size = 3 
        min_word_duration = 0.10
        i = 0
        
        while i < len(clip_segments):
            group = clip_segments[i:i + group_size]
            if len(group) < 2: 
                break
                
            group_start = group[0]["start"]
            group_end = group[-1]["end"] if i + group_size >= len(clip_segments) else min(
                group[-1]["end"], clip_segments[i + group_size]["start"])
            
            start_offset = max(0.0, group_start - clip_start)
            end_offset = max(0.0, group_end - clip_start)
            
            karaoke_text = ""
            group_duration = group_end - group_start
            
            total_natural_duration = sum(
                seg["end"] - seg["start"] for seg in group)
            
            for j, seg in enumerate(group):
                word = seg["word"]
                word_natural_duration = seg["end"] - seg["start"]
                
                if total_natural_duration > 0:
                    word_duration = max(
                        min_word_duration, (word_natural_duration / total_natural_duration) * group_duration)
                else:
                    word_duration = group_duration / len(group)
                
                word_duration_cs = int(word_duration * 100)
                
                word_upper = word.upper()
                if len(word_upper) > 10 and j > 0:
                    karaoke_text += r"\N"
                
                karaoke_text += r"{\kf" + \
                    str(word_duration_cs) + r"}" + word_upper
                
                if j < len(group) - 1 and len(word_upper) <= 10:
                    karaoke_text += " "
            
            # Get scaled font size from frontend, or use a default
            font_size = new_style.fontsize
            if captions_style_options and 'fontSize' in captions_style_options:
                font_size = int(
                    captions_style_options['fontSize'] * (subs.info["PlayResY"] / 600)) * SCALE_CORRECTION

            # ---- NEW: Dynamic karaoke styling based on ideal output ----
            # This block creates the styling seen in image #2.
            # \fnImpact: Sets the bold font.
            # \fs{font_size}: Sets the dynamic font size.
            # \c&HFFFFFF&: Primary color (white).
            # \2c&H00A5FF&: Highlight color (orange).
            # \3c&H000000&: Outline color (black).
            # \bord3\shad2: Border and shadow thickness.
            # \q2: Word-by-word karaoke.
            # \an5: Center alignment (used with \pos tag).
            styling_tags = (
                r"{\fnImpact"
                r"\fs" + str(font_size) +
                r"\c&HFFFFFF&"
                r"\2c&H00A5FF&"
                r"\3c&H000000&"
                r"\bord3\shad2\q2\an5}"
            )
            
            # Prepend the position tag to the karaoke text block
            final_text = f"{{{position_tag}}}{styling_tags}{karaoke_text}"

            subs.events.append(pysubs2.SSAEvent(
                start=pysubs2.make_time(s=start_offset),
                end=pysubs2.make_time(s=end_offset),
                text=final_text,
                style=style_name
            ))
            
            i += group_size
    else:
        # Existing group-based logic for styles 1 and 2
        group_size = 3
        min_word_duration = 0.10
        i = 0
        group_index = 0  # Track which group/set we're in
        while i < len(clip_segments):
            group = clip_segments[i:i + group_size]
            if len(group) < group_size:
                break

            # Define colors for each set (red, yellow, green)
            set_colors = [
                # Red text with subtle red glow
                (r"\c&H0000FF&", r"\3c&H4040FF&"),
                # Yellow text with subtle yellow glow
                (r"\c&H00FFFF&", r"\3c&H40FFFF&"),
                # Green text with subtle green glow
                (r"\c&H00FF00&", r"\3c&H40FF40&")
            ]
            current_set_color, current_glow_color = set_colors[group_index % 3]

            group_start = group[0]["start"]
            group_end = group[-1]["end"] if i + group_size >= len(clip_segments) else min(
                group[-1]["end"], clip_segments[i + group_size]["start"])
            group_duration = group_end - group_start
            total_min_duration = min_word_duration * group_size
            word_timings = []

            if group_duration >= total_min_duration:
                total_natural = sum(seg["end"] - seg["start"] for seg in group)
                current_time = group_start
                for j, seg in enumerate(group):
                    dur = max(
                        min_word_duration, (seg["end"] - seg["start"]) / total_natural * group_duration)
                    word_start = current_time
                    word_end = current_time + dur
                    current_time = word_end
                    word_timings.append((word_start, word_end))
            else:
                word_duration = group_duration / group_size
                word_timings = [(group_start + j * word_duration, group_start +
                                 (j + 1) * word_duration) for j in range(group_size)]

            for j in range(group_size):
                word_start, word_end = word_timings[j]
                start_offset = max(0.0, word_start - clip_start)
                end_offset = max(0.0, word_end - clip_start)
                formatted_text = ""
                for k, seg in enumerate(group):
                    word_text = seg["word"]
                    if k == j:
                        if style == 2:
                            # Current word: Use set color with reduced glow for readability
                            formatted_text += rf"{{\fs150{current_set_color}{current_glow_color}\4c&H202020&\blur1}}" + \
                                word_text.upper() + \
                                r"{\fs100\c&HFFFFFF&\3c&H606060&\4c&H202020&\blur1} "
                        else:
                            # Style 1: Blue highlighting
                            formatted_text += r"{\c&H0000FF&}" + \
                                word_text + r"{\c&HFFFFFF&} "
                    else:
                        if style == 2:
                            # Other words: White with very subtle glow
                            formatted_text += r"{\blur0.5\c&HFFFFFF&\3c&H606060&}" + \
                                word_text.upper() + " "
                        else:
                            formatted_text += word_text + " "
                
                # Prepend the position tag to the entire line of text
                final_text = f"{{{position_tag}}}{formatted_text.strip()}"

                subs.events.append(pysubs2.SSAEvent(
                    start=pysubs2.make_time(s=start_offset),
                    end=pysubs2.make_time(s=end_offset),
                    text=final_text,
                    style=style_name
                ))
            i += group_size
            group_index += 1 

    subs.save(subtitle_path)
    
    # Generate emoji overlays for styles that support them
    emoji_overlays = []
    if style in [2]:  # Only style 2 has emojis currently
        emoji_manager = ProductionEmojiManager(
            emoji_config_path, emoji_filenames_path)
        emoji_overlays = generate_emoji_overlays_production(
            clip_segments, clip_start, emoji_manager)

    # Build and execute FFmpeg command
    ffmpeg_cmd = create_optimized_ffmpeg_command_production(
        emoji_overlays, clip_video_path, subtitle_path, output_path)
    subprocess.run(ffmpeg_cmd, shell=True, check=True)


# Global emoji manager for reuse across multiple calls
_global_emoji_manager = None


def get_emoji_manager(emoji_config_path="/emoji_config.json", emoji_filenames_path="/emoji_filenames.json"):
    """
    Get or create global emoji manager instance for efficiency
    Reuses the same instance across multiple subtitle generation calls
    """
    global _global_emoji_manager
    if _global_emoji_manager is None:
        _global_emoji_manager = ProductionEmojiManager(
            emoji_config_path, emoji_filenames_path)
    return _global_emoji_manager


def generate_raw_clip(base_dir: pathlib.Path, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int) -> str:
    """
    Processes a single clip segment from a larger video.
    This function cuts the video, converts it to a vertical format,
    and uploads the subtitle-less raw clip to R2.

    Returns:
        str: The public URL of the raw vertical clip in R2.
    """
    clip_name = f"clip_{clip_index}_raw"
    s3_key_dir = os.path.dirname(s3_key)
    # The output key for the raw, subtitle-less vertical clip
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
    cut_command = (f"ffmpeg -y -i {original_video_path} -ss {start_time} -t {duration} "
                   f"-c copy {clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
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
    
    # Construct the public URL for the uploaded raw clip
    final_url = f"https://castclip.revolt-ai.com/{output_s3_key}"
    print(f"Uploaded raw clip {clip_index} to {final_url}")
    
    return final_url


def generate_raw_clip_threadsafe(args):
    """
    Thread-safe wrapper for generate_raw_clip for parallel processing.
    """
    try:
        raw_clip_url = generate_raw_clip(
            base_dir=args["base_dir"],
            original_video_path=args["video_path"],
            s3_key=args["s3_key"],
            start_time=args["start"],
            end_time=args["end"],
            clip_index=args["index"],
        )
        # Return all necessary data for the webhook
        return {
            "index": args["index"], 
            "status": "success",
            "raw_clip_url": raw_clip_url,
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
        return {"index": args["index"], "status": "error", "error": str(e)}


@app.cls(gpu="L40S", timeout=1800, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")], volumes={mount_path: volume})
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

    def _process_video_job(self, request_dict: dict, run_id: str):
        """
        Long-running processing job executed in the background.
        --- ARCHITECTURE CHANGE ---
        This job is now STAGE 1 of a two-stage process.
        1.  (This job) ANALYSIS: Transcribes video, identifies moments, cuts the
            original video into subtitle-less vertical clips, and uploads them to R2.
            It then sends a webhook with a list of these "raw" clips.
        2.  (render_final_clip job) RENDERING: A separate, fast CPU job is called
            by the frontend to burn subtitles onto a raw clip after user edits.
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
                    ids = request.ids
                    s3_key = f"{ids}/input.mp4"
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

            # 3. Process Clips into RAW vertical videos (no subtitles)
            clip_args_list = []
            for index, moment in enumerate(clip_moments[:clip_count]):
                if "start" in moment and "end" in moment:
                    duration = moment["end"] - moment["start"]
                    if duration >= 10:
                        hook = moment.get("hook", "")
                        
                        # Filter the full transcript to get segments only for this clip
                        clip_specific_transcript = [
                            s for s in transcript_segments 
                            if s.get("start", 0) >= moment["start"] and s.get("end", 0) <= moment["end"]
                        ]

                        print(
                            f"Queueing raw clip {index}: Hook = '{hook}', Duration = {duration:.1f}s")
                        clip_args_list.append({
                            "base_dir": base_dir,
                            "video_path": str(video_path),
                            "s3_key": s3_key,
                            "start": moment["start"],
                            "end": moment["end"],
                            "index": index,
                            "transcript_segments": clip_specific_transcript,  # Pass only relevant transcript
                            "hook": hook,
                        })
            
            # Optimized threading with dynamic worker count
            # Limit workers to avoid overwhelming system
            max_workers = min(len(clip_args_list), 4)
            print(
                f"Processing {len(clip_args_list)} raw clips using {max_workers} threads...")
            
            final_clips_data = []
            if clip_args_list:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(
                        generate_raw_clip_threadsafe, clip_args_list))
                
                successful_clips = [
                    r for r in results if r["status"] == "success"]
                failed_clips = [r for r in results if r["status"] == "error"]
                
                print(
                    f"🎉 Raw clip processing complete: {len(successful_clips)} successful, {len(failed_clips)} failed")
                
                if failed_clips:
                    for failed_clip in failed_clips:
                        print(
                            f"❌ Failed raw clip {failed_clip['index']}: {failed_clip.get('error', 'Unknown error')}")
                
                # Prepare successful clip data for the webhook
                if successful_clips:
                    # Sort by index to maintain order
                    successful_clips.sort(key=lambda x: x['index'])
                    for clip_result in successful_clips:
                        final_clips_data.append({
                            "raw_clip_url": clip_result["raw_clip_url"],
                            "transcript_segments": clip_result["transcript_segments"],
                            "hook": clip_result["hook"],
                            "start": clip_result["start"],
                            "end": clip_result["end"]
                        })

            if request.webhook_url:
                user_id, project_id = request.ids.split(
                    "/", 1) if "/" in request.ids else (request.ids, "default")
                
                if final_clips_data:
                    # Send a list of raw clips ready for user review and editing
                    webhook_status = "ready_for_review"
                    webhook_error_message = None
                else:
                    # Handle case where no valid clips could be generated
                    webhook_status = "failed"
                    webhook_error_message = "No valid clips could be generated from the video."

                webhook_success = send_completion_webhook(
                    webhook_url=request.webhook_url,
                    user_id=user_id,
                    project_id=project_id,
                    status=webhook_status,
                    clips=final_clips_data,
                    error_message=webhook_error_message
                )
                if not webhook_success:
                    logging.warning(
                        f"Failed to send '{webhook_status}' webhook for project {project_id}")
            else:
                print("No webhook URL provided. Skipping notification.")

        except Exception as e:
            print(f"Unhandled error during process_video: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error webhook if processing failed
            if hasattr(request, 'webhook_url') and request.webhook_url:
                try:
                    user_id, project_id = request.ids.split(
                        "/", 1) if "/" in request.ids else (request.ids, "default")
                    send_completion_webhook(
                        webhook_url=request.webhook_url,
                        user_id=user_id,
                        project_id=project_id,
                        status="failed",
                        error_message=str(e)
                    )
                except Exception as webhook_error:
                    logging.error(
                        f"Failed to send error webhook: {webhook_error}")
            
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
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
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
        
        **Clip Length**: Aim to generate clips that are between 20 and 120 seconds long. Include as much relevant content as possible within this timeframe. IF a specific story goes beyond the 120 second limit, you are allowed to exceed it to make sure the entire story is told, but this is only in special cases, the HARD LIMIT is 150 seconds
       
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
        max_retries = 2
        clip_count_per_chunk = max(1, clip_count // len(valid_transcripts))
        results = []

        for chunk in valid_transcripts:
            prompt = self.get_prompt(chunk, clip_count_per_chunk)
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemini-2.5-pro",
                        config=types.GenerateContentConfig(safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                        ],
                            system_instruction="You are a Podcast Clip Extractor, tasked with creating clips for short-form platforms like TikTok and YouTube Shorts. Your goal is to identify and extract engaging stories, questions, and answers from podcast transcripts that will appeal to up-and-coming entrepreneurs seeking advice and motivation."),
                        contents=[
                            types.Content(
                                role="user",
                                parts=[types.Part(text=prompt)]
                            ),
                            types.Part.from_bytes(
                                data=pdf_data,
                                mime_type="application/pdf"
                            )
                        ]
                    )

                    if response.text:
                        results.append(response.text)
                        break
                    else:
                        print(
                            f"Attempt {attempt}: Empty response.text from Gemini")

                except Exception as e:
                    print(f"Attempt {attempt}: Gemini API call failed - {e}")

                print("Retrying...")
                time.sleep(2)

        return results if results else "[]"

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, background_tasks: BackgroundTasks, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        # Authenticate early and respond fast for async processing
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect Bearer Token", headers={"WWW-Authenticate": "Bearer"})

        # Generate a job id and enqueue background work
        run_id = str(uuid.uuid4())
        background_tasks.add_task(
            self._process_video_job, request.model_dump(), run_id)

        # Immediate 202 so Inngest doesn't block on long GPU work
        return JSONResponse(status_code=202, content={
            "job_id": run_id,
            "status": "accepted",
            "message": "Analysis started. You will receive a webhook when clips are ready for review."
        })

    # ---- NEW: FastAPI endpoint to trigger the final rendering job ----
    # The frontend will call this API route when a user clicks "Export".
    # It authenticates the request and synchronously calls the Modal function above.
    @modal.fastapi_endpoint(method="POST")
    def render_video(self, request: RenderVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect Bearer Token", headers={"WWW-Authenticate": "Bearer"})

        print(f"Received render request for clip_id: {request.clip_id}")

        # Construct the output S3 key from user and project IDs
        try:
            path_parts = urlparse(request.raw_clip_url).path.strip('/').split('/')
            if len(path_parts) >= 3:
                user_id = path_parts[0]
                project_id = path_parts[1]
                output_s3_key = f"{user_id}/{project_id}/{request.clip_id}.mp4"
            else:
                raise HTTPException(status_code=400, detail="Invalid raw_clip_url format")
        except Exception:
            raise HTTPException(status_code=400, detail="Could not parse user_id and project_id from raw_clip_url")

        # Create a dictionary for the render_final_clip function
        render_payload = request.model_dump()
        render_payload['output_s3_key'] = output_s3_key

        # Call the Modal function and wait for it to complete.
        result = render_final_clip.remote(render_payload)

        if result["status"] == "success":
            return JSONResponse(status_code=200, content=result)
        else:
            return JSONResponse(status_code=500, content=result)


# This is STAGE 2 of the process. It's a lightweight, fast-starting CPU-only
# function designed to be called directly by the frontend when the user clicks "Export".
# It takes the raw clip and the user's final edits, burns the subtitles, and saves it.
@app.function(
    cpu=4,
    memory=2048,
    timeout=600,
    # Keep 1 container warm to eliminate cold starts for a responsive user experience
    min_containers=1,
    secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")]
)
def render_final_clip(request_dict: dict):

    try:
        # The output_s3_key is in the raw dictionary but not the Pydantic model
        # for validation purposes. We extract it here.
        output_s3_key = request_dict.get("output_s3_key")
        if not output_s3_key:
            raise ValueError("output_s3_key is missing from the request payload")
            
        request = RenderVideoRequest(**request_dict)
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download the raw vertical clip from R2
        raw_clip_path = base_dir / "raw_clip.mp4"
        print(f"Downloading raw clip from {request.raw_clip_url}")
        with requests.get(request.raw_clip_url, stream=True) as r:
            r.raise_for_status()
            with open(raw_clip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # 2. Burn subtitles using the provided (and possibly edited) data
        final_video_path = base_dir / "final_video.mp4"
        print("Burning final subtitles...")
        create_subtitles_with_ffmpeg(
            transcript_segments=[seg.model_dump()
                                 for seg in request.transcript_segments],
            clip_start=0,  # The clip is already cut, so we process from the beginning
            clip_end=None,  # Process until the end of the clip
            clip_video_path=str(raw_clip_path),
            output_path=str(final_video_path),
            hook=request.hook,
            style=request.project_style,
            hook_style_options=request.hook_style.model_dump(
            ) if request.hook_style else None,
            captions_style_options=request.captions_style.model_dump() if request.captions_style else None
        )

        # 3. Upload the final rendered video to R2
        print(f"Uploading final video to S3 key: {output_s3_key}")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            region_name="auto"
        )
        s3_client.upload_file(
            str(final_video_path), "ai-podcast-clipper", output_s3_key,
            ExtraArgs={
                'ContentType': 'video/mp4',
                'ContentDisposition': f'attachment; filename="{os.path.basename(output_s3_key)}"'
            }
        )

        # 4. Clean up local files
        shutil.rmtree(base_dir)

        # 5. Return the public URL of the final video
        final_url = f"https://castclip.revolt-ai.com/{output_s3_key}"
        print(f"Successfully rendered clip: {final_url}")
        return {"status": "success", "final_clip_url": final_url}

    except Exception as e:
        print(f"❌ Error during final rendering: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    url = ai_podcast_clipper.render_video.web_url

    payload = {
        "clip_id": "52f59dea-c1ca-4a9f-8862-cef75ab21c7e",
        "raw_clip_url": "https://castclip.revolt-ai.com/3zhHBEDMtMIEiRshOHkFSV72wKfdKkKI/554ca925-5cde-4440-9399-b917c866c648/clip_0_raw.mp4",
        "transcript_segments": [
            {
                "word": "Yeah,",
                "start": 0,
                "end": 0.36099999999999
            },
            {
                "word": "so",
                "start": 0.3809999999999718,
                "end": 0.6610000000000014
            },
            {
                "word": "currently",
                "start": 1.702999999999975,
                "end": 2.202999999999975
            },
            {
                "word": "I",
                "start": 2.563999999999965,
                "end": 2.72399999999999
            },
            {
                "word": "have",
                "start": 3.08499999999998,
                "end": 3.305000000000007
            },
            {
                "word": "a",
                "start": 3.504999999999995,
                "end": 3.524999999999977
            },
            {
                "word": "lot",
                "start": 3.586000000000013,
                "end": 3.786000000000001
            },
            {
                "word": "of",
                "start": 3.825999999999965,
                "end": 3.865999999999985
            },
            {
                "word": "debt",
                "start": 4.34699999999998,
                "end": 4.586999999999989
            },
            {
                "word": "and",
                "start": 5.087999999999965,
                "end": 5.228000000000009
            },
            {
                "word": "I",
                "start": 5.788999999999987,
                "end": 5.829000000000008
            },
            {
                "word": "am",
                "start": 5.908999999999992,
                "end": 6.810999999999979
            },
            {
                "word": "currently",
                "start": 6.911000000000001,
                "end": 7.230999999999995
            },
            {
                "word": "living",
                "start": 7.290999999999997,
                "end": 7.531999999999982
            },
            {
                "word": "out",
                "start": 7.611999999999966,
                "end": 7.692000000000007
            },
            {
                "word": "of",
                "start": 7.731999999999971,
                "end": 7.771999999999991
            },
            {
                "word": "my",
                "start": 7.891999999999996,
                "end": 8.031999999999982
            },
            {
                "word": "leased",
                "start": 8.113,
                "end": 8.37299999999999
            },
            {
                "word": "car.",
                "start": 8.452999999999975,
                "end": 8.694000000000017
            },
            {
                "word": "I",
                "start": 9.213999999999999,
                "end": 9.254000000000019
            },
            {
                "word": "just",
                "start": 9.293999999999983,
                "end": 9.394999999999982
            },
            {
                "word": "threw",
                "start": 9.41500000000002,
                "end": 9.59499999999997
            },
            {
                "word": "a",
                "start": 9.634999999999991,
                "end": 9.654999999999973
            },
            {
                "word": "match.",
                "start": 9.675000000000011,
                "end": 10.87700000000001
            },
            {
                "word": "You",
                "start": 10.89699999999999,
                "end": 11.49799999999999
            },
            {
                "word": "have",
                "start": 11.51799999999997,
                "end": 11.59800000000001
            },
            {
                "word": "a",
                "start": 11.61799999999999,
                "end": 11.63799999999998
            },
            {
                "word": "leased",
                "start": 11.65800000000002,
                "end": 11.798
            },
            {
                "word": "car?",
                "start": 11.81799999999998,
                "end": 11.89799999999997
            },
            {
                "word": "Okay.",
                "start": 11.91899999999998,
                "end": 12.01900000000001
            },
            {
                "word": "Yeah.",
                "start": 12.03899999999999,
                "end": 12.339
            },
            {
                "word": "So",
                "start": 12.35899999999998,
                "end": 12.399
            },
            {
                "word": "you're",
                "start": 12.43899999999996,
                "end": 12.59999999999997
            },
            {
                "word": "currently",
                "start": 12.62,
                "end": 12.92000000000002
            },
            {
                "word": "homeless?",
                "start": 12.95999999999998,
                "end": 13.32099999999997
            },
            {
                "word": "Yeah,",
                "start": 13.74099999999999,
                "end": 13.96199999999999
            },
            {
                "word": "currently",
                "start": 14.322,
                "end": 14.56299999999999
            },
            {
                "word": "homeless.",
                "start": 14.58299999999997,
                "end": 16.08499999999998
            },
            {
                "word": "Okay.",
                "start": 17.48699999999997,
                "end": 17.58699999999999
            },
            {
                "word": "Yeah.",
                "start": 18.00799999999998,
                "end": 18.28800000000001
            },
            {
                "word": "I'm",
                "start": 18.38900000000001,
                "end": 18.50900000000001
            },
            {
                "word": "sorry.",
                "start": 18.589,
                "end": 18.709
            },
            {
                "word": "Yep,",
                "start": 19.73500000000001,
                "end": 20.31599999999997
            },
            {
                "word": "it's",
                "start": 20.33600000000001,
                "end": 20.43599999999998
            },
            {
                "word": "okay.",
                "start": 20.49599999999998,
                "end": 20.77600000000001
            },
            {
                "word": "I",
                "start": 21.23699999999997,
                "end": 21.27699999999999
            },
            {
                "word": "mean,",
                "start": 21.29699999999997,
                "end": 21.517
            },
            {
                "word": "I'm-",
                "start": 21.798,
                "end": 21.95799999999997
            },
            {
                "word": "Well,",
                "start": 22.399,
                "end": 22.53899999999999
            },
            {
                "word": "how",
                "start": 22.55899999999997,
                "end": 22.63900000000001
            },
            {
                "word": "long",
                "start": 22.69900000000001,
                "end": 22.85899999999998
            },
            {
                "word": "have",
                "start": 22.91899999999998,
                "end": 23
            },
            {
                "word": "you",
                "start": 23.01999999999998,
                "end": 23.12
            },
            {
                "word": "been",
                "start": 23.15999999999997,
                "end": 23.38
            },
            {
                "word": "homeless?",
                "start": 23.38,
                "end": 23.70099999999996
            },
            {
                "word": "It's",
                "start": 23.70099999999996,
                "end": 23.80099999999999
            },
            {
                "word": "only",
                "start": 23.86099999999999,
                "end": 24.00099999999998
            },
            {
                "word": "been",
                "start": 24.041,
                "end": 24.50200000000001
            },
            {
                "word": "like",
                "start": 25.12299999999999,
                "end": 25.50299999999999
            },
            {
                "word": "three",
                "start": 25.72300000000001,
                "end": 25.88400000000001
            },
            {
                "word": "weeks.",
                "start": 25.92399999999998,
                "end": 26.12399999999997
            },
            {
                "word": "Wow,",
                "start": 27.125,
                "end": 27.38599999999997
            },
            {
                "word": "what",
                "start": 27.44599999999997,
                "end": 27.58600000000001
            },
            {
                "word": "happened",
                "start": 27.60599999999999,
                "end": 27.82599999999996
            },
            {
                "word": "three",
                "start": 27.846,
                "end": 27.98599999999999
            },
            {
                "word": "weeks",
                "start": 28.02699999999999,
                "end": 28.20699999999999
            },
            {
                "word": "ago?",
                "start": 28.28699999999998,
                "end": 28.50700000000001
            },
            {
                "word": "So",
                "start": 29.649,
                "end": 29.72899999999998
            },
            {
                "word": "what",
                "start": 29.74899999999997,
                "end": 29.82900000000001
            },
            {
                "word": "happened",
                "start": 29.84899999999999,
                "end": 30.00900000000001
            },
            {
                "word": "three",
                "start": 30.029,
                "end": 30.12900000000002
            },
            {
                "word": "weeks",
                "start": 30.14999999999998,
                "end": 30.32999999999998
            },
            {
                "word": "ago",
                "start": 30.34999999999997,
                "end": 30.55000000000001
            },
            {
                "word": "is",
                "start": 30.56999999999999,
                "end": 30.61000000000001
            },
            {
                "word": "the",
                "start": 30.971,
                "end": 31.03100000000001
            },
            {
                "word": "guys",
                "start": 33.17399999999998,
                "end": 33.49399999999997
            },
            {
                "word": "whose",
                "start": 33.57400000000001,
                "end": 33.89499999999998
            },
            {
                "word": "room",
                "start": 33.97499999999997,
                "end": 34.09499999999997
            },
            {
                "word": "I",
                "start": 34.11500000000001,
                "end": 34.13499999999999
            },
            {
                "word": "was",
                "start": 34.17500000000001,
                "end": 34.23500000000001
            },
            {
                "word": "renting",
                "start": 34.255,
                "end": 34.476
            },
            {
                "word": "out,",
                "start": 34.55599999999998,
                "end": 34.67599999999999
            },
            {
                "word": "he",
                "start": 35.577,
                "end": 35.637
            },
            {
                "word": "ended",
                "start": 35.67699999999996,
                "end": 35.81700000000001
            },
            {
                "word": "up",
                "start": 35.87700000000001,
                "end": 35.93799999999999
            },
            {
                "word": "selling",
                "start": 35.95799999999997,
                "end": 36.21800000000002
            },
            {
                "word": "his",
                "start": 36.25799999999998,
                "end": 36.33799999999997
            },
            {
                "word": "property",
                "start": 36.37799999999999,
                "end": 36.71899999999999
            },
            {
                "word": "and",
                "start": 36.75900000000001,
                "end": 37.45999999999998
            },
            {
                "word": "kicked",
                "start": 37.65999999999997,
                "end": 37.81999999999999
            },
            {
                "word": "me",
                "start": 37.83999999999997,
                "end": 37.89999999999998
            },
            {
                "word": "out.",
                "start": 38,
                "end": 38.14100000000002
            },
            {
                "word": "And",
                "start": 38.36099999999999,
                "end": 38.44099999999997
            },
            {
                "word": "because",
                "start": 38.48099999999999,
                "end": 38.86199999999997
            },
            {
                "word": "I",
                "start": 38.88200000000001,
                "end": 38.90199999999999
            },
            {
                "word": "have",
                "start": 39.88299999999998,
                "end": 40.16300000000001
            },
            {
                "word": "no",
                "start": 40.20400000000001,
                "end": 40.72399999999999
            },
            {
                "word": "saved",
                "start": 41.10500000000002,
                "end": 41.36500000000001
            },
            {
                "word": "money,",
                "start": 41.44499999999999,
                "end": 41.64600000000002
            },
            {
                "word": "I",
                "start": 41.666,
                "end": 41.68599999999998
            },
            {
                "word": "couldn't",
                "start": 42.06599999999997,
                "end": 42.24599999999998
            },
            {
                "word": "even",
                "start": 42.286,
                "end": 42.42699999999996
            },
            {
                "word": "afford",
                "start": 42.46699999999998,
                "end": 42.80700000000002
            },
            {
                "word": "a",
                "start": 42.84699999999998,
                "end": 42.86700000000002
            },
            {
                "word": "down",
                "start": 43.06799999999998,
                "end": 43.32799999999997
            },
            {
                "word": "payment",
                "start": 43.36799999999999,
                "end": 43.72800000000001
            },
            {
                "word": "for",
                "start": 44.32900000000001,
                "end": 44.44900000000001
            },
            {
                "word": "another",
                "start": 44.50999999999999,
                "end": 44.70999999999998
            },
            {
                "word": "place.",
                "start": 44.84999999999997,
                "end": 45.59100000000001
            }
        ],
        "hook": "mason is so nice and loving",
        "hook_style": {
            "fontSize": 30.15317295942923,
            "position": {
                "x": 48.85714285714286,
                "y": 18.94641153131435
            }
        },
        "project_style": 3,
        "captions_style": {
            "fontSize": 48.40808127191877,
            "position": {
                "x": 48.285714285714285,
                "y": 74.50002511174725
            }
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer DhfvXdZywkp8PTdAYjQLrBbgBbdSztfiqZPGXdeyY96msKe6gmJXZz93fGkGUsmh"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
