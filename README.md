# AI Podcast Clipper

A production-ready AI-powered tool that automatically creates short, engaging clips from long-form podcast videos for social media platforms like TikTok, Instagram Reels, and YouTube Shorts.

## Features

- **Multi-source input**: Process videos from both S3 storage and YouTube URLs
- **AI-powered moment detection**: Uses Google Gemini to identify the most engaging moments
- **Automatic transcription**: High-quality transcription with word-level timestamps using WhisperX
- **Speaker tracking**: Intelligent face tracking and cropping for vertical video format
- **Subtitle generation**: Automatic subtitle overlay with customizable styling
- **Scalable processing**: Built on Modal for GPU-accelerated processing

## Quick Start

### YouTube URL Processing
```python
import requests

payload = {
    "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "clip_count": 3
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_TOKEN"
}

response = requests.post("YOUR_MODAL_ENDPOINT", json=payload, headers=headers)
result = response.json()
```

### S3 Video Processing
```python
import requests

payload = {
    "s3_key": "path/to/your/video.mp4",
    "clip_count": 2
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_TOKEN"
}

response = requests.post("YOUR_MODAL_ENDPOINT", json=payload, headers=headers)
result = response.json()
```

## API Reference

### Endpoint: POST /process_video

**Request Body:**
```json
{
    "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID",  // Optional
    "s3_key": "path/to/video.mp4",                            // Optional
    "clip_count": 3                                           // Required (1-10)
}
```

**Note**: Provide either `youtube_url` OR `s3_key`, not both.

**Response:**
```json
{
    "run_id": "uuid-string",
    "video_source": "youtube|s3",
    "processed_clips": [
        {
            "clip_index": 0,
            "start_time": 45.2,
            "end_time": 78.9,
            "duration": 33.7,
            "s3_key": "output/path/clip_0.mp4",
            "status": "success"
        }
    ],
    "total_clips_processed": 2,
    "total_clips_requested": 3,
    "video_info": {  // Only for YouTube videos
        "title": "Video Title",
        "duration": 3600,
        "uploader": "Channel Name",
        "upload_date": "20231201",
        "view_count": 1000000
    }
}
```

## Video Requirements

### YouTube Videos
- **Duration**: 1 minute to 3 hours
- **Format**: Any format supported by yt-dlp
- **Quality**: Automatically downloads best quality up to 1080p

### S3 Videos
- **Format**: MP4 recommended
- **Duration**: No specific limits (reasonable processing time expected)

## Configuration

### Environment Variables
```bash
# Required
AUTH_TOKEN=your_auth_token
GEMINI_API_KEY=your_gemini_api_key

# For S3/R2 storage
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_ENDPOINT_URL=your_r2_endpoint
```

### Modal Secrets
Create a Modal secret named `ai-podcast-clipper-secret` with the environment variables above.

## Processing Pipeline

1. **Video Download**: Downloads from YouTube or S3
2. **Transcription**: WhisperX generates word-level transcripts
3. **Moment Identification**: Gemini AI finds engaging Q&A segments or stories
4. **Video Processing**: Creates vertical format with speaker tracking
5. **Subtitle Addition**: Adds styled subtitles to final clips
6. **Upload**: Stores processed clips in S3

## Output Format

- **Aspect Ratio**: 9:16 (vertical for mobile)
- **Resolution**: 1080x1920
- **Codec**: H.264
- **Audio**: AAC, 128k bitrate
- **Subtitles**: Embedded with custom styling

## Error Handling

The API provides detailed error responses:

- **400 Bad Request**: Invalid YouTube URL, video too long/short, download failures
- **401 Unauthorized**: Invalid authentication token
- **500 Internal Server Error**: Processing failures with detailed error messages

## Deployment

1. Install Modal CLI: `pip install modal`
2. Set up Modal account and authenticate
3. Create the required secret with environment variables
4. Deploy: `modal deploy main.py`

## Development

### Local Testing
```bash
# Run the local entrypoint
modal run main.py
```

### Dependencies
All dependencies are managed in `requirements.txt` and automatically installed in the Modal container.

## Production Considerations

- **Rate Limiting**: Consider implementing rate limiting for YouTube downloads
- **Content Policy**: Ensure compliance with YouTube's Terms of Service
- **Storage Costs**: Monitor S3 storage usage for clips
- **Processing Limits**: 3-hour maximum for YouTube videos
- **Concurrent Processing**: Modal handles scaling automatically

## License

[Your License Here]

## Support

For issues and feature requests, please create an issue in the repository. 