import os
import subprocess
import json
import pysubs2


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
    filters = ["[0:v]null[v0]"]
    current_output = "[v0]"

    # Group overlays by URL for efficiency
    url_overlays = {}
    for overlay in emoji_overlays:
        url = overlay["url"]
        if url not in url_overlays:
            url_overlays[url] = []
        url_overlays[url].append(overlay)

    filter_idx = 1
    for url, overlays in url_overlays.items():
        input_idx = unique_urls[url]

        # Create time conditions for this emoji
        time_conditions = []
        for overlay in overlays:
            time_conditions.append(
                f"between(t,{overlay['start']},{overlay['end']})")

        enable_condition = "+".join(time_conditions)
        next_output = f"[v{filter_idx}]"

        filters.append(
            f"{current_output}[{input_idx}:v] overlay=190:500:enable='{enable_condition}' {next_output}")
        current_output = next_output
        filter_idx += 1

    # Add subtitles last
    filters.append(f"{current_output} ass={subtitle_path} [final]")

    filter_chain = "; ".join(filters)
    return f'ffmpeg -y {" ".join(inputs)} -filter_complex "{filter_chain}" -map "[final]" -map 0:a? -c:v h264 -preset fast -crf 23 "{output_path}"'


def create_subtitles_with_ffmpeg(transcript_segments, clip_start, clip_end, clip_video_path, output_path,
                                 style=1, emoji_config_path="emoji_config.json",
                                 emoji_filenames_path="emoji_filenames.json"):
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
        new_style.fontname = "Impact"
        new_style.fontsize = 100
        new_style.primarycolor = pysubs2.Color(255, 255, 255)
        new_style.outline = 2.0
        new_style.shadow = 0.0
        new_style.outlinecolor = pysubs2.Color(100, 100, 100)
        new_style.blur = 1.0
        new_style.alignment = 2
        new_style.marginl = 40
        new_style.marginr = 40
        new_style.marginv = 800  # Position higher for emoji above
        new_style.spacing = 2.0

    elif style == 3:
        # Style 3: TODO - Future implementation
        raise NotImplementedError("Style 3 not implemented yet")

    elif style == 4:
        # Style 4: TODO - Future implementation
        raise NotImplementedError("Style 4 not implemented yet")

    subs.styles[style_name] = new_style

    # Generate subtitle events with word-level timing
    group_size = 3
    min_word_duration = 0.10
    i = 0
    group_index = 0

    while i < len(clip_segments):
        group = clip_segments[i:i + group_size]
        if len(group) < group_size:
            break

        # Define colors for each set (red, yellow, green)
        set_colors = [
            (r"\c&H0000FF&", r"\3c&H4040FF&"),  # Red text with subtle red glow
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

        # Calculate word timings
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

        # Create subtitle events for each word in the group
        for j in range(group_size):
            word_start, word_end = word_timings[j]
            start_offset = max(0.0, word_start - clip_start)
            end_offset = max(0.0, word_end - clip_start)
            formatted_text = ""

            for k, seg in enumerate(group):
                word_text = seg["word"]
                if k == j:
                    # Current word formatting
                    if style == 2:
                        formatted_text += rf"{{\fs150{current_set_color}{current_glow_color}\4c&H202020&\blur1}}" + \
                            word_text.upper() + \
                            r"{\fs100\c&HFFFFFF&\3c&H606060&\4c&H202020&\blur1} "
                    else:
                        formatted_text += r"{\c&H0000FF&}" + \
                            word_text + r"{\c&HFFFFFF&} "
                else:
                    # Other words formatting
                    if style == 2:
                        formatted_text += r"{\blur0.5\c&HFFFFFF&\3c&H606060&}" + \
                            word_text.upper() + " "
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

    # Save subtitle file
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


def get_emoji_manager(emoji_config_path="emoji_config.json", emoji_filenames_path="emoji_filenames.json"):
    """
    Get or create global emoji manager instance for efficiency
    Reuses the same instance across multiple subtitle generation calls
    """
    global _global_emoji_manager
    if _global_emoji_manager is None:
        _global_emoji_manager = ProductionEmojiManager(
            emoji_config_path, emoji_filenames_path)
    return _global_emoji_manager


def main():
    """
    Example usage of the subtitle generator for style 2.
    This function creates necessary dummy files, runs the subtitle generation,
    and cleans up afterwards.
    """
    print("🎬 Starting subtitle generation example for Style 2...")

    # --- 1. Setup required files and data ---

    # Dummy transcript data
    transcript_segments = [
        {"start": 0.031, "end": 0.191, "word": "tables"},
        {"start": 0.211, "end": 0.371, "word": "turn,"},
        {"start": 0.391, "end": 0.531, "word": "bridges"},
        {"start": 0.551, "end": 0.671, "word": "burn"},
        {"start": 0.691, "end": 0.831, "word": "you"},
        {"start": 0.851, "end": 0.971, "word": "live"},
        {"start": 0.991, "end": 1.111, "word": "and"},
        {"start": 1.131, "end": 1.271, "word": "you"},
        {"start": 1.291, "end": 1.431, "word": "learn"}
    ]

    # Use the actual emoji config files
    emoji_config_path = "emoji_config.json"
    emoji_filenames_path = "emoji_filenames.json"

    # Create a dummy video file
    input_video_path = "dummy_input.mp4"
    output_video_path = "output_style2.mp4"
    
    # Check if ffmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("🔴 FFmpeg is not installed or not in PATH. Please install FFmpeg to run this example.")
        return

    print(f"🛠️ Creating dummy video: {input_video_path}")
    ffmpeg_create_dummy_cmd = (
        f'ffmpeg -y -f lavfi -i color=c=black:s=1080x1920:r=30:d=3 '
        f'-c:v libx264 -pix_fmt yuv420p "{input_video_path}"'
    )
    subprocess.run(ffmpeg_create_dummy_cmd, shell=True, check=True, capture_output=True, text=True)
    
    # --- 2. Run the subtitle generation ---
    
    print("🔥 Generating subtitles with emojis (Style 2)...")
    try:
        create_subtitles_with_ffmpeg(
            transcript_segments=transcript_segments,
            clip_start=0,
            clip_end=2,
            clip_video_path=input_video_path,
            output_path=output_video_path,
            style=2,
            emoji_config_path=emoji_config_path,
            emoji_filenames_path=emoji_filenames_path
        )
        print(f"✅ Successfully created video: {output_video_path}")
    except Exception as e:
        print(f"❌ An error occurred during subtitle generation: {e}")

    # --- 3. Clean up temporary files ---

    print("🧹 Cleaning up temporary files...")
    temp_files = [
        input_video_path,
        "temp_subtitles.ass"  # Created internally by the function
    ]
    for f_path in temp_files:
        if os.path.exists(f_path):
            os.remove(f_path)
            print(f"   - Removed {f_path}")

    print("🎉 Example run finished.")


if __name__ == "__main__":
    main()
