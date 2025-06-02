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

class ProcessVideoRequest(BaseModel):
    s3_key: str
    clip_count: int = 1


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                #    "wget -O /usr/share/fonts/truetype/custom/FjallaOne-Regular.ttf https://github.com/google/fonts/raw/main/ofl/fjallaone/FjallaOne-Regular.ttf",
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

def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [s for s in transcript_segments
                     if s.get("start") and s.get("end") and s["end"] > clip_start and s["start"] < clip_end]

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
    new_style.primarycolor = pysubs2.Color(255, 255, 255)  # white
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 600
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    group_size = 3
    min_word_duration = 0.2  # Minimum duration for each word highlight
    
    i = 0
    while i < len(clip_segments):
        group = clip_segments[i:i + group_size]
        if len(group) < group_size:
            break

        # Calculate seamless timing for the entire group
        group_start = group[0]["start"]
        
        # Determine group end - extend to next group start if gap is small
        if i + group_size < len(clip_segments):
            next_group_start = clip_segments[i + group_size]["start"]
            natural_group_end = group[-1]["end"]
            gap_to_next = next_group_start - natural_group_end
            
            # If gap is small (≤500ms), extend to next group start
            if gap_to_next <= 0.5:
                group_end = next_group_start
            else:
                group_end = natural_group_end
        else:
            group_end = group[-1]["end"]
        
        # Calculate seamless word timings within the group
        group_duration = group_end - group_start
        word_timings = []
        
        # Ensure each word gets at least min_word_duration
        total_min_duration = min_word_duration * group_size
        
        if group_duration >= total_min_duration:
            # Distribute time proportionally but ensure minimums
            natural_durations = [seg["end"] - seg["start"] for seg in group]
            total_natural = sum(natural_durations)
            
            current_time = group_start
            for j, seg in enumerate(group):
                if j == group_size - 1:
                    # Last word gets remaining time
                    word_start = current_time
                    word_end = group_end
                else:
                    # Calculate proportional duration but enforce minimum
                    natural_duration = seg["end"] - seg["start"]
                    proportional_duration = (natural_duration / total_natural) * group_duration
                    actual_duration = max(min_word_duration, proportional_duration)
                    
                    word_start = current_time
                    word_end = current_time + actual_duration
                    current_time = word_end
                
                word_timings.append((word_start, word_end))
        else:
            # Group duration is too short, divide equally
            word_duration = group_duration / group_size
            for j in range(group_size):
                word_start = group_start + (j * word_duration)
                word_end = group_start + ((j + 1) * word_duration)
                word_timings.append((word_start, word_end))

        # Create subtitle events with seamless timing
        for j in range(group_size):
            word_start, word_end = word_timings[j]
            
            start_offset = max(0.0, word_start - clip_start)
            end_offset = max(0.0, word_end - clip_start)

            # Create the formatted text with highlighting
            formatted_text = ""
            for k, seg in enumerate(group):
                word_text = seg["word"]
                if k == j:
                    formatted_text += r"{\c&H0000FF&}" + word_text + r"{\c&HFFFFFF&} "
                else:
                    formatted_text += word_text + " "

            subs.events.append(pysubs2.SSAEvent(
                start=pysubs2.make_time(s=start_offset),
                end=pysubs2.make_time(s=end_offset),
                text=formatted_text.strip(),
                style=style_name
            ))

        i += group_size

    subs.save(subtitle_path)

    ffmpeg_cmd = f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" -c:v h264 -preset fast -crf 23 {output_path}"
    subprocess.run(ffmpeg_cmd, shell=True, check=True)

def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
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
    
    create_subtitles_with_ffmpeg(transcript_segments, start_time, end_time, vertical_mp4_path, subtitle_output_path)
    
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

    s3_client.upload_file(final_output_path, "ai-podcast-clipper", output_s3_key)

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
        prompt = f""" 
                You will be provided with a podcast transcript, including the start and end times for each word:

Follow these steps to create informative clips that will help and provide value to the user:

1.  **Analyze the Transcript**: Carefully review the entirety of the provided podcast transcript to identify potential stories, questions, and answers that would be valuable and engaging for up-and-coming entrepreneurs.
2.  **Identify Attention-Grabbing Hooks**: Prioritize clips that have an attention-grabbing hook within the first 3 seconds to maximize viewer engagement. This could be a surprising statement, a controversial opinion, or a compelling question. This is VERY IMPORTANT!
3.  **Select Relevant Segments**: Choose segments that include a question and its corresponding answer. It is acceptable to include a few additional sentences before the question to provide context.
4.  **Adhere to Timestamps**: Use ONLY the start and end timestamps provided in the input. Do NOT modify the timestamps. The start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
5.  **Ensure Non-Overlapping Clips**: Make sure that the selected clips do not overlap with one another. Each clip should be a distinct segment of the podcast.
6.  **Clip Length**: Aim to generate clips that are between 20 and 60 seconds long. Include as much relevant content as possible within this timeframe. The user has requested you make {clip_count} clips.
7.  **Exclude Irrelevant Content**: Avoid including the following in your clips:

 * Moments of greeting, thanking, or saying goodbye.
 * Non-question and answer interactions or segments that do not provide value to the audience.
8.  **Prioritize Value and Knowledge**: Ensure that the selected clips provide tangible value and knowledge to the viewer. The clips should not only be attention-grabbing but also offer meaningful insights, practical advice, or valuable information that the audience can learn from.
9. **Include the following**: Hook within the first 3 seconds (controversial opinion, bold claim, mystery), Standalone context (doesn’t require earlier podcast sections to understand), Emotional impact (funny, shocking, inspiring, relatable).
10. **Format Output as JSON**: Format your output as a list of JSON objects, with each object representing a clip and containing 'start' and 'end' timestamps in seconds, and a 'reason' explaining why this clip was chosen and why it will perform well on TikTok. The output should be readable by Python's `json.loads` function.

```json
[
{{\"start\": seconds, \"end\": seconds, \"reason\": \"Explanation for TikTok performance\"}},
...clip2,
clip3
]
```
10. **Handle No Valid Clips**: If there are no valid clips to extract based on the above criteria, output an empty list in JSON format: `[]`. This output should also be readable by `json.loads()` in Python.

Example:

If the podcast transcript contains a segment where a question is asked at 60 seconds and answered by 80 seconds, and it meets all the criteria above, the output would be:

```json
[
{{\"start\": 60, \"end\": 80, \"reason\": \"This clip contains a surprising statistic that will grab viewers' attention on TikTok.\"}}
]
``` 
Use the provided file and these guidelines to create clips from the entire podcast

Here is the transcript:
{transcript}
                """
        pdf_url = "https://castclip.revolt-ai.com/app/Why%20clips%20went%20viral.pdf"
        pdf_data = httpx.get(pdf_url).content
        max_retries = 2
        for attempt in range (1, max_retries + 1):
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-05-20", 
                    config=types.GenerateContentConfig(safety_settings=[
                        types.SafetySetting( category="HARM_CATEGORY_HARASSMENT", threshold="OFF"), 
                        types.SafetySetting( category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF" ), 
                        types.SafetySetting( category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF" ), 
                        types.SafetySetting( category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"), 
                    ],
                    system_instruction="You are a Podcast Clip Extractor, tasked with creating promotional clips for short-form platforms like TikTok and YouTube Shorts. Your goal is to identify and extract engaging stories, questions, and answers from podcast transcripts that will appeal to up-and-coming entrepreneurs seeking advice and motivation. To help you create the best possible clips, you will be given a pdf file that will contain the transcript of 20 other clips and details about what made them go viral. Use this to guide you in choosing what parts to clip from the provided transcript"),
                    contents= [
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
                    return response.text
                
                print(f"Attempt {attempt}: Empty response.text from Gemini")

            except Exception as e:
                print(f"Attempt {attempt}: Gemini API call failed - {e}")
            
            print("Retrying...")
            time.sleep(2)

        
        print("All Gemini retires failed.")
        return "[]"


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
                s3_key = f"{uuid.uuid4()}/input.mp4"  # Synthetic s3_key placeholder for naming consistency
            else:
                print("Downloading video from S3...")
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                    endpoint_url=os.environ["R2_ENDPOINT_URL"],
                    region_name="auto"
                )
                s3_client.download_file("ai-podcast-clipper", s3_key, str(video_path))

            
            # 1. Transcription
            transcript_segments_json = self.transcribe_video(base_dir, video_path)
            transcript_segments = json.loads(transcript_segments_json)
            
            # 2. Identify moments for clips
            print("Identifying clip moments")
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
            
            print(clip_moments)

            # 3. Process Clips
            for index, moment in enumerate(clip_moments[:clip_count]):
                if "start" in moment and "end" in moment:
                    duration = moment["end"] - moment["start"]
                    if duration >= 10: 
                        print("Processing clip" + str(index) + " from " +
                            str(moment["start"]) + " to " + str(moment["end"]))
                        process_clip(base_dir, video_path, s3_key, moment["start"], moment["end"], index, transcript_segments)  
      
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
        "s3_key": "3zhHBEDMtMIEiRshOHkFSV72wKfdKkKI/1c15fb4c-388d-481f-b0c4-d96573289d4a/drake5min.mp4",
        "clip_count": 1
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(url, json=payload,
                             headers=headers)
    
    # with open("output.json", "w", encoding="utf-8") as f:
    #     f.write(json.dumps(response.json(), indent=2))

    response.raise_for_status()