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
import pysubs2
from google.genai import types
from yt_dlp import YoutubeDL
import httpx
import concurrent.futures


class ProcessVideoRequest(BaseModel):
    s3_key: str
    clip_count: int = 1
    style: int = 1


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
    .add_local_file("emoji_filenames.json", "/emoji_filenames.json"))

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
        print(f"🚀 Production Emoji Manager: {len(self.word_to_filename)} mappings ready")
    
    def _load_configurations(self, emoji_config_path, emoji_filenames_path):
        """Load and cross-reference emoji configs with available filenames"""
        
        # Load available filenames from R2
        try:
            with open(emoji_filenames_path, 'r', encoding='utf-8') as f:
                filenames_list = json.load(f)
                self.available_filenames = set(filenames_list)
                print(f"✅ Loaded {len(self.available_filenames)} available emoji files")
        except FileNotFoundError:
            print(f"⚠️ {emoji_filenames_path} not found. Using emoji config only.")
            self.available_filenames = set()
        
        # Load word-to-emoji mappings
        try:
            with open(emoji_config_path, 'r', encoding='utf-8') as f:
                emoji_data = json.load(f)
                self._create_word_to_filename_mapping(emoji_data)
        except FileNotFoundError:
            print(f"⚠️ {emoji_config_path} not found. Using minimal fallback mapping.")
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
        
        print(f"📊 Mapping Stats: {mapped_count} words mapped, {unmapped_count} words unmapped")
    
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
            group_end = group[-1]["end"] if i + group_size >= len(clip_segments) else min(group[-1]["end"], clip_segments[i + group_size]["start"])
            
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
            time_conditions.append(f"between(t,{overlay['start']},{overlay['end']})")
        
        enable_condition = "+".join(time_conditions)
        next_output = f"[v{filter_idx}]"
        
        # Position adjusted to be above text and centered
        filters.append(f"{current_output}{scaled_emoji_stream} overlay=(main_w-overlay_w)/2:1200:enable='{enable_condition}' {next_output}")
        current_output = next_output
        filter_idx += 1
    
    # Add subtitles last
    filters.append(f"{current_output} ass={subtitle_path} [final]")
    
    filter_chain = "; ".join(filters)
    return f'ffmpeg -y {" ".join(inputs)} -filter_complex "{filter_chain}" -map "[final]" -map 0:a? -c:v h264 -preset fast -crf 23 "{output_path}"'


def create_subtitles_with_ffmpeg(transcript_segments, clip_start, clip_end, clip_video_path, output_path, 
                                style=1, emoji_config_path="/emoji_config.json", 
                                emoji_filenames_path="/emoji_filenames.json"):
    """
    Create subtitles with optional emoji overlays using FFmpeg
    
    Args:
        transcript_segments: List of transcript segments with word timing
        clip_start: Start time of the clip in seconds
        clip_end: End time of the clip in seconds (None for full video)
        clip_video_path: Input video file path
        output_path: Output video file path
        style: Subtitle style (1-4, currently 1 and 2 implemented)
        emoji_config_path: Path to emoji configuration JSON
        emoji_filenames_path: Path to emoji filenames JSON
    
    Raises:
        ValueError: If style is not between 1 and 4
        FileNotFoundError: If required video file doesn't exist
    """
    
    if style not in [1, 2, 3, 4]:
        raise ValueError("Style must be between 1 and 4")
    
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

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    
    # Configure style based on parameter
    if style == 1:
        # Style 1: Regular subtitles (Anton font, no emojis)
        new_style.fontname = "Anton"
        new_style.fontsize = 140
        new_style.primarycolor = pysubs2.Color(255, 255, 255)
        new_style.outline = 2.0
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
    
    subs.styles[style_name] = new_style

    if style == 3:
        group_size = 3 
        min_word_duration = 0.10
        i = 0
        
        while i < len(clip_segments):
            group = clip_segments[i:i + group_size]
            if len(group) < 2: 
                break
                
            group_start = group[0]["start"]
            group_end = group[-1]["end"] if i + group_size >= len(clip_segments) else min(group[-1]["end"], clip_segments[i + group_size]["start"])
            
            start_offset = max(0.0, group_start - clip_start)
            end_offset = max(0.0, group_end - clip_start)
            
            karaoke_text = ""
            group_duration = group_end - group_start
            
            total_natural_duration = sum(seg["end"] - seg["start"] for seg in group)
            
            for j, seg in enumerate(group):
                word = seg["word"]
                word_natural_duration = seg["end"] - seg["start"]
                
                if total_natural_duration > 0:
                    word_duration = max(min_word_duration, (word_natural_duration / total_natural_duration) * group_duration)
                else:
                    word_duration = group_duration / len(group)
                
                word_duration_cs = int(word_duration * 100)
                
                word_upper = word.upper()
                if len(word_upper) > 10 and j > 0:
                    karaoke_text += r"\N"
                
                karaoke_text += r"{\kf" + str(word_duration_cs) + r"}" + word_upper
                
                if j < len(group) - 1 and len(word_upper) <= 10:
                    karaoke_text += " "
            
            full_karaoke_text = (
                r"{\fn" + r"Impact" + r"\fs100\b1}" +  
                r"{\c&HFFFFFF&\2c&H00A5FF&\3c&H000000&\4c&H303030&" + 
                r"\blur2.5\bord3\shad2}" +  
                r"{\q3\an2}" +  
                karaoke_text
            )
            
            subs.events.append(pysubs2.SSAEvent(
                start=pysubs2.make_time(s=start_offset),
                end=pysubs2.make_time(s=end_offset),
                text=full_karaoke_text,
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
                (r"\c&H0000FF&", r"\3c&H4040FF&"),  # Red text with subtle red glow
                (r"\c&H00FFFF&", r"\3c&H40FFFF&"),  # Yellow text with subtle yellow glow  
                (r"\c&H00FF00&", r"\3c&H40FF40&")   # Green text with subtle green glow
            ]
            current_set_color, current_glow_color = set_colors[group_index % 3]

            group_start = group[0]["start"]
            group_end = group[-1]["end"] if i + group_size >= len(clip_segments) else min(group[-1]["end"], clip_segments[i + group_size]["start"])
            group_duration = group_end - group_start
            total_min_duration = min_word_duration * group_size
            word_timings = []

            if group_duration >= total_min_duration:
                total_natural = sum(seg["end"] - seg["start"] for seg in group)
                current_time = group_start
                for j, seg in enumerate(group):
                    dur = max(min_word_duration, (seg["end"] - seg["start"]) / total_natural * group_duration)
                    word_start = current_time
                    word_end = current_time + dur
                    current_time = word_end
                    word_timings.append((word_start, word_end))
            else:
                word_duration = group_duration / group_size
                word_timings = [(group_start + j * word_duration, group_start + (j + 1) * word_duration) for j in range(group_size)]

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
                            formatted_text += rf"{{\fs150{current_set_color}{current_glow_color}\4c&H202020&\blur1}}" + word_text.upper() + r"{\fs100\c&HFFFFFF&\3c&H606060&\4c&H202020&\blur1} "
                        else:
                            # Style 1: Blue highlighting
                            formatted_text += r"{\c&H0000FF&}" + word_text + r"{\c&HFFFFFF&} "
                    else:
                        if style == 2:
                            # Other words: White with very subtle glow
                            formatted_text += r"{\blur0.5\c&HFFFFFF&\3c&H606060&}" + word_text.upper() + " "
                        else:
                            formatted_text += word_text + " "
                subs.events.append(pysubs2.SSAEvent(
                    start=pysubs2.make_time(s=start_offset),
                    end=pysubs2.make_time(s=end_offset),
                    text=formatted_text.strip(),
                    style=style_name
                ))
            i += group_size
            group_index += 1 

    subs.save(subtitle_path)
    
    # Generate emoji overlays for styles that support them
    emoji_overlays = []
    if style in [2]:  # Only style 2 has emojis currently
        emoji_manager = ProductionEmojiManager(emoji_config_path, emoji_filenames_path)
        emoji_overlays = generate_emoji_overlays_production(clip_segments, clip_start, emoji_manager)

    # Build and execute FFmpeg command
    ffmpeg_cmd = create_optimized_ffmpeg_command_production(emoji_overlays, clip_video_path, subtitle_path, output_path)
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
        _global_emoji_manager = ProductionEmojiManager(emoji_config_path, emoji_filenames_path)
    return _global_emoji_manager


def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list, style: int = 1):
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
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

    create_subtitles_with_ffmpeg(
        transcript_segments, start_time, end_time, vertical_mp4_path, subtitle_output_path, style=style)

    # Final TikTok-friendly normalization
    final_output_path = clip_dir / "pyavi" / "final_output.mp4"
    ffmpeg_final_cmd = (
        f"ffmpeg -y -i {subtitle_output_path} "
        f"-c:v libx264 -preset slow -crf 23 "
        f"-c:a aac -b:a 128k -movflags +faststart "
        f"-pix_fmt yuv420p "
        f"-metadata title='Podcast Clip' "
        f"-metadata creation_time='{time.strftime('%Y-%m-%dT%H:%M:%S')}' "
        f"{final_output_path}"
    )
    subprocess.run(ffmpeg_final_cmd, shell=True, check=True)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        region_name="auto"  # Cloudflare R2 ignores region, but boto3 still requires it
    )

    s3_client.upload_file(
        final_output_path, "ai-podcast-clipper", output_s3_key,
        ExtraArgs={
            'ContentType': 'video/mp4',
            'ContentDisposition': f'attachment; filename="{os.path.basename(output_s3_key)}"',
            'Metadata': {
                'clip-index': str(clip_index),
                'processed-time': time.strftime('%Y-%m-%dT%H:%M:%S')
            }
        }
    )
    

def generate_clip_threadsafe(args):
    try:
        process_clip(
            base_dir=args["base_dir"],
            original_video_path=args["video_path"],
            s3_key=args["s3_key"],
            start_time=args["start"],
            end_time=args["end"],
            clip_index=args["index"],
            transcript_segments=args["transcript_segments"],
            style=args["style"]
        )
        return {"index": args["index"], "status": "success"}
    except Exception as e:
        return {"index": args["index"], "status": "error", "error": str(e)}


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")], volumes={mount_path: volume})
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

    def download_youtube_video(self, youtube_url: str, output_path: pathlib.Path):
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': str(output_path),
            'merge_output_format': 'mp4',
            'quiet': True,
            'noplaylist': True,
            'nocheckcertificate': True,
            'retries': 3,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                return info.get("title", "Untitled Video")
        except Exception as e:
            raise RuntimeError(f"Failed to download YouTube video: {e}")

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
                Here is a podcast transcript, including the start and end times for each word:
                {json.dumps(transcript, indent=2)}
                Follow these steps to create informative clips that will help and provide value to the user:

                1. **Analyze the Transcript**: Carefully review the entirety of the provided podcast transcript to identify engaging stories, and anecdotes that would be valuable and engaging for up-and-coming entrepreneurs.
                2. **Identify Attention-Grabbing Hooks**: Prioritize clips that have an attention-grabbing hook within the first 3 seconds to maximize viewer engagement. This could be a surprising statement, a controversial opinion,
                or a compelling question. This is VERY IMPORTANT! This hook will be displayed as text on the video exactly how you give it so make sure it really forces the viewer to want to watch the clip all the way through. The max
                length of a hook is 7 words. To help you in writing hooks I have uploaded a file that serves as a guide to writing hooks for videos on tiktok. Another thing I want you to look out for when writing hooks for clips is to 
                make sure that the hook makes viewers want to watch the entire video and doesn't dismiss them too soon. For example if you have a hook like "You dramatically underestimate this one thing." and the clip starts with the 
                speaker saying what the thing that "you" underestimate is in the first 5 seconds, then the viewer won't be incentivized to watch the whole thing. Keep this in mind when choosing the start and end times for clips and creating hooks.
                3. **Select Relevant Segments**: Choose segments that include engaging stories or a question and answer, where the clip starts with a question that upon hearing the user will want to continue watching to know the answer. It is acceptable to include a few additional sentences before the story to provide context. 
                4. **Adhere to Timestamps**: Use ONLY the start and end timestamps provided in the input. Do NOT modify the timestamps. The start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
                5. **Ensure Non-Overlapping Clips**: Make sure that the selected clips do not overlap with one another. Each clip should be a distinct segment of the podcast.
                6. **Clip Length**: Aim to generate clips that are between 20 and 60 seconds long. Include as much relevant content as possible within this timeframe. The user has requested you make 
                {clip_count} clips. IF a specific story goes beyond the 60 second limit, you are allowed to exceed it to make sure the entire story is told.
                7. **Exclude Irrelevant Content**: Avoid including the following in your clips:

                * Moments of greeting, thanking, or saying goodbye. This includes introductions that may be good hooks but don't actually provide value at the end. (eg. usually end with "let's get into it")
                * Interactions or segments that do not provide value to the audience.
                8. **Prioritize Value and Knowledge**: Ensure that the selected clips provide tangible value and knowledge to the viewer. The clips should not only be attention-grabbing but also offer meaningful insights, practical advice, or valuable information that the audience can learn from.
                9. **Include the following**: Hook within the first 3 seconds (controversial opinion, bold claim, mystery), Standalone context (doesn't require earlier podcast sections to understand), Emotional impact (funny, shocking, inspiring, relatable).
                10. **Format Output as JSON**: Format your output as a list of JSON objects, with each object representing a clip and containing 'start' and 'end' timestamps in seconds, and a 'hook' to help captivate the viewer as soon as they read it to incentivize them to watch the whole clip. The output should be readable by Python's `json.loads` function.

                ```json
                [
                {{"start": seconds, "end": seconds, "hook": "something to capture the viewers attention"}},
                ...clip2,
                clip3
                ]
                ```
                10. **Handle No Valid Clips**: If there are no valid clips to extract based on the above criteria, output an empty list in JSON format: `[]`. This output should also be readable by `json.loads()` in Python.

                Use the provided file and these guidelines to create clips from the entire podcast
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
                model="gemini-2.5-pro-preview-06-05",
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
                            system_instruction="You are a Podcast Clip Extractor, tasked with creating  clips for short-form platforms like TikTok and YouTube Shorts. Your goal is to identify and extract engaging stories, questions, and answers from podcast transcripts that will appeal to up-and-coming entrepreneurs seeking advice and motivation."),
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
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        try:
            s3_key = request.s3_key
            clip_count = request.clip_count

            if token.credentials != os.environ["AUTH_TOKEN"]:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                    detail="Incorrect Bearer Token", headers={"WWW-Authenticate": "Bearer"})

            run_id = str(uuid.uuid4())
            base_dir = pathlib.Path("/tmp") / run_id
            base_dir.mkdir(parents=True, exist_ok=True)

            # Download Video
            video_path = base_dir / "input.mp4"

            if request.s3_key.startswith("http://") or request.s3_key.startswith("https://"):
                print("Detected YouTube URL, starting download...")
                self.download_youtube_video(request.s3_key, video_path)
                # Synthetic s3_key placeholder for naming consistency
                s3_key = f"{uuid.uuid4()}/input.mp4"
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
            # return transcript_segments

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


            clip_args_list = []
            # 3. Process Clips
            for index, moment in enumerate(clip_moments[:clip_count]):
                if "start" in moment and "end" in moment:
                    duration = moment["end"] - moment["start"]
                    if duration >= 10:
                        clip_args_list.append({
                            "base_dir": base_dir,
                            "video_path": video_path,
                            "s3_key": s3_key,
                            "start": moment["start"],
                            "end": moment["end"],
                            "index": index,
                            "transcript_segments": transcript_segments,
                            "style": request.style
                        })
            
            # Run clips in parallel using threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                list(executor.map(generate_clip_threadsafe, clip_args_list))

        except Exception as e:
            print(f"Unhandled error during process_video: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            if base_dir.exists():
                print(f"Cleaning up temp dir after {base_dir}")
                shutil.rmtree(base_dir, ignore_errors=True)


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    url = ai_podcast_clipper.process_video.web_url

    payload = {
        "s3_key": "3zhHBEDMtMIEiRshOHkFSV72wKfdKkKI/162e8d57-008e-47ec-b3b1-6f8614aabf85/video.mp4",
        "clip_count": 3,
        "style": 3
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer DhfvXdZywkp8PTdAYjQLrBbgBbdSztfiqZPGXdeyY96msKe6gmJXZz93fGkGUsmh"
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    # Save the transcript response to output.json
    # transcript_data = response.json()
    # with open("output_2.json", "w", encoding="utf-8") as f:
    #     json.dump(transcript_data, f, indent=2, ensure_ascii=False)
    
    # print("Transcript saved to output.json")
      
