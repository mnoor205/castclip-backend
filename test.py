import os
import subprocess
import json
from PIL import Image, ImageDraw, ImageFont
import pysubs2
import re

# Paste or import your transcript_segments here
transcript_segments = [{'start': 0.031, 'end': 0.292, 'word': 'Sun'}, {'start': 0.332, 'end': 0.472, 'word': "don't"}, {'start': 0.492, 'end': 0.672, 'word': 'stop'}, {'start': 0.712, 'end': 0.772, 'word': 'for'}, {'start': 0.813, 'end': 1.073, 'word': 'nobody,'}, {'start': 1.093, 'end': 1.253, 'word': 'man.'}, {'start': 2.656, 'end': 2.796, 'word': 'Sun'}, {'start': 2.816, 'end': 2.957, 'word': "don't"}, {'start': 2.977, 'end': 3.197, 'word': 'fucking'}, {'start': 3.217, 'end': 3.458, 'word': 'stop.'}, {'start': 3.518, 'end': 3.678, 'word': 'Sun'}, {'start': 3.698, 'end': 3.838, 'word': 'gonna'}, {'start': 3.878, 'end': 3.959, 'word': 'be'}, {'start': 4.019, 'end': 4.059, 'word': 'up'}, {'start': 4.099, 'end': 4.139, 'word': 'in'}, {'start': 4.159, 'end': 4.219, 'word': 'the'}, {'start': 4.259, 'end': 4.46, 'word': 'morning'}, {'start': 4.5, 'end': 4.94, 'word': 'regardless.'}, {'start': 6.043, 'end': 6.183, 'word': 'That'}, {'start': 6.223, 'end': 6.483, 'word': 'sun'}, {'start': 6.604, 'end': 6.664, 'word': 'is'}, {'start': 6.684, 'end': 6.884, 'word': 'gonna'}, {'start': 6.964, 'end': 7.245, 'word': 'be'}, {'start': 7.706, 'end': 7.826, 'word': 'up'}, {'start': 8.407, 'end': 8.467, 'word': 'in'}, {'start': 8.507, 'end': 8.567, 'word': 'the'}, {'start': 8.587, 'end': 8.808, 'word': 'morning'}, {'start': 8.828, 'end': 9.149, 'word': 'regardless.'}, {'start': 9.169, 'end': 9.409, 'word': 'Regardless'}, {'start': 9.429, 'end': 9.509, 'word': 'of'}, {'start': 9.549, 'end': 9.71, 'word': 'how'}, {'start': 9.79, 'end': 9.89, 'word': 'I'}, {'start': 9.93, 'end': 10.271, 'word': 'fucking'}, {'start': 10.331, 'end': 10.611, 'word': 'feel'}, {'start': 10.651, 'end': 11.493, 'word': 'and'}, {'start': 11.533, 'end': 11.653, 'word': 'how'}, {'start': 11.713, 'end': 12.074, 'word': 'depressed'}, {'start': 12.174, 'end': 12.234, 'word': 'I'}, {'start': 12.335, 'end': 12.435, 'word': 'am,'}, {'start': 12.475, 'end': 12.615, 'word': 'the'}, {'start': 12.775, 'end': 13.096, 'word': 'sun'}, {'start': 13.136, 'end': 13.938, 'word': 'is'}, {'start': 13.978, 'end': 14.178, 'word': 'gonna'}, {'start': 14.198, 'end': 14.539, 'word': 'fucking'}, {'start': 14.579, 'end': 14.859, 'word': 'shine'}, {'start': 14.88, 'end': 14.92, 'word': 'in'}, {'start': 14.96, 'end': 15.02, 'word': 'the'}, {'start': 15.06, 'end': 15.36, 'word': 'morning'}, {
    'start': 16.402, 'end': 16.483, 'word': 'and'}, {'start': 16.523, 'end': 16.563, 'word': 'at'}, {'start': 16.623, 'end': 17.004, 'word': 'nighttime'}, {'start': 17.064, 'end': 17.124, 'word': 'the'}, {'start': 17.184, 'end': 17.384, 'word': 'moon'}, {'start': 17.404, 'end': 17.585, 'word': 'gonna'}, {'start': 17.625, 'end': 17.725, 'word': 'be'}, {'start': 17.765, 'end': 17.885, 'word': 'there'}, {'start': 17.905, 'end': 17.965, 'word': 'and'}, {'start': 18.006, 'end': 18.066, 'word': 'you'}, {'start': 18.126, 'end': 18.226, 'word': 'gonna'}, {'start': 18.246, 'end': 18.386, 'word': 'look'}, {'start': 18.446, 'end': 18.527, 'word': 'up,'}, {'start': 18.567, 'end': 18.687, 'word': 'these'}, {'start': 18.747, 'end': 18.987, 'word': 'days'}, {'start': 19.027, 'end': 19.168, 'word': 'gonna'}, {'start': 19.208, 'end': 19.348, 'word': 'keep'}, {'start': 19.388, 'end': 19.589, 'word': 'going'}, {'start': 19.609, 'end': 19.789, 'word': 'by.'}, {'start': 21.452, 'end': 21.552, 'word': 'So'}, {'start': 21.572, 'end': 21.632, 'word': 'do'}, {'start': 21.673, 'end': 21.753, 'word': 'you'}, {'start': 21.793, 'end': 21.893, 'word': 'let'}, {'start': 21.933, 'end': 21.993, 'word': 'the'}, {'start': 22.033, 'end': 22.234, 'word': 'days'}, {'start': 22.294, 'end': 22.414, 'word': 'go'}, {'start': 22.474, 'end': 22.634, 'word': 'by'}, {'start': 22.674, 'end': 22.735, 'word': 'and'}, {'start': 22.775, 'end': 22.915, 'word': 'look'}, {'start': 22.975, 'end': 23.035, 'word': 'up'}, {'start': 23.095, 'end': 23.155, 'word': 'and'}, {'start': 23.175, 'end': 23.236, 'word': 'you'}, {'start': 23.256, 'end': 23.376, 'word': 'done'}, {'start': 23.396, 'end': 23.656, 'word': 'wasted'}, {'start': 23.696, 'end': 23.716, 'word': 'a'}, {'start': 23.757, 'end': 23.917, 'word': 'year'}, {'start': 24.037, 'end': 24.278, 'word': 'doing'}, {'start': 24.378, 'end': 24.598, 'word': 'what?'}, {'start': 25.8, 'end': 25.861, 'word': 'Or'}, {'start': 25.881, 'end': 25.921, 'word': 'do'}, {'start': 25.941, 'end': 26.001, 'word': 'you'}, {'start': 26.021, 'end': 26.141, 'word': 'just'}, {'start': 26.201, 'end': 26.342, 'word': 'pick'}, {'start': 26.382, 'end': 26.422, 'word': 'it'}, {'start': 26.522, 'end': 26.602, 'word': 'up?'}, {'start': 27.965, 'end': 28.025, 'word': 'All'}, {'start': 28.045, 'end': 28.165, 'word': 'right.'}]


# Set the start and end time for the subtitles (full video by default)
clip_start = 0.0
clip_end = None  # Will be set to video duration if None
clip_video_path = "clips/clip.mp4"
output_path = "clip_with_subs.mp4"

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
        
        print(f"Word to filename mapping: {self.word_to_filename}")
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
            "🚨": "police_car_light.png",
            "🌉": "bridge_at_night.png"
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
        word_clean = re.sub(r'[^\w\s]', '', word.lower().strip())
        filename = self.word_to_filename.get(word_clean)
        print(f"Getting emoji for: '{word}' -> '{word_clean}', found: {filename}")
        
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

# Initialize production emoji manager
emoji_manager = ProductionEmojiManager()

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
    
    print(f"Generated emoji overlays: {emoji_overlays}")
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
        filters.append(f"[{input_idx}:v]scale=100:100{scaled_stream_name}")
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
        filters.append(f"{current_output}{scaled_emoji_stream} overlay=(main_w-overlay_w)/2:800:enable='{enable_condition}' {next_output}")
        current_output = next_output
        filter_idx += 1
    
    # Add subtitles last
    filters.append(f"{current_output} ass={subtitle_path} [final]")
    
    filter_chain = "; ".join(filters)
    final_command = f'ffmpeg -y {" ".join(inputs)} -filter_complex "{filter_chain}" -map "[final]" -map 0:a? -c:v h264 -preset fast -crf 23 "{output_path}"'
    print(f"Generated FFmpeg command:\n{final_command}")
    return final_command

def create_subtitles_with_ffmpeg(transcript_segments, clip_start, clip_end, clip_video_path, output_path, style=1, title=None):
    if style not in [1, 2, 3, 4]:
        raise ValueError("Style must be between 1 and 4")
    
    temp_dir = os.path.dirname(output_path) or "."
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    if clip_end is None:
        import ffmpeg
        clip_end = float(ffmpeg.probe(clip_video_path)["format"]["duration"])

    clip_segments = [s for s in transcript_segments if s.get("start") and s.get("end") and s["end"] > clip_start and s["start"] < clip_end]

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
        # Style 2: Different font with emojis and subtle glow effect
        new_style.fontname = "impact"  # Different fontS
        new_style.fontsize = 100      # Base font size
        new_style.primarycolor = pysubs2.Color(255, 255, 255)  # White text
        new_style.outline = 2.0       # Reduced outline for less glow
        new_style.shadow = 0.0        # No shadow, we'll use blur for glow
        new_style.outlinecolor = pysubs2.Color(100, 100, 100)  # Subtle gray outline
        new_style.blur = 1.0          # Reduced blur for subtle glow effect
        new_style.alignment = 2
        new_style.marginl = 40
        new_style.marginr = 40
        new_style.marginv = 400  # Position higher to leave room for emoji above
        new_style.spacing = 2.0
    elif style == 3:
        # Style 3: Karaoke style 
        new_style.fontname = "Impact"  
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
        # Style 4: Placeholder for future implementation
        print("Style 4 not implemented yet")
        return
    
    subs.styles[style_name] = new_style

    # Add title style if title is provided
    if title:
        title_style = pysubs2.SSAStyle()
        title_style.fontname = "Impact"
        title_style.fontsize = 93
        title_style.primarycolor = pysubs2.Color(255, 255, 255)  # White text
        title_style.outline = 3.0
        title_style.outlinecolor = pysubs2.Color(0, 0, 0)  # Black outline
        title_style.shadow = 2.0
        title_style.shadowcolor = pysubs2.Color(0, 0, 0, 180)  # Dark shadow
        title_style.alignment = 8  # Center-top alignment (not center-bottom)
        title_style.marginl = 60
        title_style.marginr = 60
        title_style.marginv = 250  # Distance from top edge
        title_style.bold = True
        title_style.spacing = 1.0
        
        subs.styles["Title"] = title_style
        
        # Add title event that spans the entire video
        title_text = title.upper()

        subs.events.append(pysubs2.SSAEvent(
            start=pysubs2.make_time(s=0),
            end=pysubs2.make_time(s=clip_end - clip_start if clip_end else 30),
            text=title_text,
            style="Title"
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
            group_index += 1  # Move to next set

    subs.save(subtitle_path)
    
    # Only generate emoji overlays for styles that support them
    emoji_overlays = []
    if style in [2]:  # Only style 2 has emojis for now
        emoji_overlays = generate_emoji_overlays_production(clip_segments, clip_start, emoji_manager)

    # Build FFmpeg command based on whether we have emoji overlays
    ffmpeg_cmd = create_optimized_ffmpeg_command_production(emoji_overlays, clip_video_path, subtitle_path, output_path)
    
    subprocess.run(ffmpeg_cmd, shell=True, check=True)


if __name__ == "__main__":
    if not transcript_segments:
        print("Please provide transcript_segments as a list of dicts with 'start', 'end', and 'word'.")
    else:
        # Style 1: Regular subtitles with Anton font, blue highlighting
        # Style 2: Impact font with emojis and color cycling
        # Style 3: Karaoke style with word-by-word sweeping highlight effects
        # Style 4: Not implemented yet
        
        # Test with title - similar to the style shown in the image
        create_subtitles_with_ffmpeg(
            transcript_segments, 
            clip_start, 
            clip_end, 
            clip_video_path, 
            "clip_with_subs_style_1.mp4", 
            style=1, 
            title="Josh Allen made a core memory for this fan"
        )
