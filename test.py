import os
import subprocess
import pysubs2
from typing import List, Dict

def create_advanced_subtitles_with_title(transcript_segments: List[Dict], clip_video_path: str, output_path: str, title: str = "Best piece of advice"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subtitle_path = os.path.join(os.path.dirname(output_path), "temp_subtitles.ass")
    
    # Set video duration
    import cv2
    video = cv2.VideoCapture(clip_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    video.release()
    
    # Create ASS file
    subs = pysubs2.SSAFile()
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"
    
    # Define styles
    title_style = pysubs2.SSAStyle(
    fontname="Arial Black",  # Or try "Montserrat Black" if installed
    fontsize=78,
    primarycolor=pysubs2.Color(0, 0, 0),   # Black
    outline=0,
    shadow=0,
    alignment=8,  # Top center
    marginl=60,
    marginr=60,
    marginv=30,   # Closer to top
    borderstyle=4,  # Box
    backcolor=pysubs2.Color(255, 255, 255, 0),  # White, fully opaque
    bold=True
    )
    subs.styles["Title"] = title_style

    # Pad with spaces for a thicker "pill"
    title_box_text = f"  {title}  "
    subs.events.append(pysubs2.SSAEvent(
        start=0, end=pysubs2.make_time(s=duration), text=title_box_text, style="Title"
    ))

    default_style = pysubs2.SSAStyle(
        fontname="SofiaSansCondensed[wght]", fontsize=56, primarycolor=pysubs2.Color(255,255,255),
        outline=3, shadow=0, outlinecolor=pysubs2.Color(0,0,0), alignment=2,
        marginl=40, marginr=40, marginv=200
    )
    subs.styles["Default"] = default_style

    highlight_style = pysubs2.SSAStyle(
        fontname="SofiaSansCondensed[wght]", fontsize=56, primarycolor=pysubs2.Color(255,255,255),
        outline=0, shadow=0, alignment=2, marginl=40, marginr=40, marginv=200,
        borderstyle=4, backcolor=pysubs2.Color(0,0,255,0)
    )
    subs.styles["Highlight"] = highlight_style

    # Group words into phrases (4 words per phrase as in original)
    phrase_groups = []
    current_phrase = []
    for segment in transcript_segments:
        word = segment.get("word", "").strip()
        if not word:
            continue
        start_rel = segment.get("start", 0)
        end_rel = segment.get("end", 0)
        if end_rel <= 0:
            continue
        current_phrase.append({'word': word, 'start': start_rel, 'end': end_rel})
        if len(current_phrase) >= 4 or (len(current_phrase) >= 2 and word.endswith(('.', '!', '?', ','))):
            phrase_groups.append(current_phrase)
            current_phrase = []
    if current_phrase:
        phrase_groups.append(current_phrase)
    
    # Create subtitle events for each phrase with word-by-word highlighting
    for phrase_group in phrase_groups:
        if not phrase_group:
            continue
        for i, word_info in enumerate(phrase_group):
            word_start = word_info['start']
            word_end = word_info['end']
            subtitle_parts = []
            for j, w in enumerate(phrase_group):
                if j == i:
                    subtitle_parts.append(f"{{\\rHighlight}}{w['word']}{{\\rDefault}}")
                else:
                    subtitle_parts.append(w['word'])
            subtitle_text = ' '.join(subtitle_parts)
            event = pysubs2.SSAEvent(
                start=pysubs2.make_time(s=word_start),
                end=pysubs2.make_time(s=word_end),
                text=subtitle_text,
                style="Default"
            )
            subs.events.append(event)
    # Save the ASS file
    subs.save(subtitle_path)

    subtitle_path = os.path.abspath(subtitle_path).replace("\\", "/")
    # For Windows: Escape colon after drive letter (C:/ becomes C\:/)
    import re
    subtitle_path_ffmpeg = re.sub(r"^([A-Za-z]):/", r"\1\\:/", subtitle_path)

    ffmpeg_cmd = (
        f'ffmpeg -y -i "{clip_video_path}" '
        f'-vf "ass=\'{subtitle_path_ffmpeg}\'" '
        f'-c:v h264 -preset fast -crf 23 '
        f'-c:a copy '
        f'"{output_path}"'
    )
    print("Running:", ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd, shell=True, check=True)


if __name__ == "__main__":
    import json

    # === INPUTS ===
    INPUT_VIDEO = "videos/drake10sec.mp4"       # Path to your cropped video file
    TRANSCRIPT_JSON = "transcript_output.json"   # Path to transcript file (see below)
    OUTPUT_DIR = "clips"
    OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "final_output.mp4")
    TITLE = "Best piece of advice"

    # Load transcript array
    with open(TRANSCRIPT_JSON, "r", encoding="utf-8") as f:
        transcript_segments = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_advanced_subtitles_with_title(
        transcript_segments, INPUT_VIDEO, OUTPUT_VIDEO, title=TITLE
    )
    print(f"Done! Video saved to {OUTPUT_VIDEO}")
