#!/usr/bin/env python3
"""
Example usage script for AI Podcast Clipper

This script demonstrates how to use the AI Podcast Clipper with both
YouTube URLs and S3 keys.
"""

import requests
import json
import time
from typing import Dict, Any


class PodcastClipperClient:
    def __init__(self, endpoint_url: str, auth_token: str):
        self.endpoint_url = endpoint_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}"
        }
    
    def process_youtube_video(self, youtube_url: str, clip_count: int = 3) -> Dict[str, Any]:
        """Process a YouTube video to create clips."""
        payload = {
            "youtube_url": youtube_url,
            "clip_count": clip_count
        }
        return self._make_request(payload)
    
    def process_s3_video(self, s3_key: str, clip_count: int = 3) -> Dict[str, Any]:
        """Process an S3 video to create clips."""
        payload = {
            "s3_key": s3_key,
            "clip_count": clip_count
        }
        return self._make_request(payload)
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual request to the API."""
        print(f"Sending request to: {self.endpoint_url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = requests.post(
                self.endpoint_url, 
                json=payload, 
                headers=self.headers,
                timeout=900  # 15 minutes timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if e.response.content:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    print(f"Error details: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            raise
    
    def print_results(self, result: Dict[str, Any]):
        """Pretty print the processing results."""
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        
        print(f"Run ID: {result.get('run_id')}")
        print(f"Video Source: {result.get('video_source')}")
        print(f"Clips Processed: {result.get('total_clips_processed')}/{result.get('total_clips_requested')}")
        
        if result.get('video_source') == 'youtube':
            video_info = result.get('video_info', {})
            print(f"\nVideo Information:")
            print(f"  Title: {video_info.get('title')}")
            print(f"  Duration: {video_info.get('duration')}s ({video_info.get('duration', 0)/60:.1f} minutes)")
            print(f"  Uploader: {video_info.get('uploader')}")
            print(f"  Upload Date: {video_info.get('upload_date')}")
            print(f"  View Count: {video_info.get('view_count', 0):,}")
        
        print(f"\nProcessed Clips:")
        for clip in result.get('processed_clips', []):
            status_icon = "✓" if clip['status'] == 'success' else "✗"
            print(f"  {status_icon} Clip {clip['clip_index']}: {clip['start_time']:.1f}s - {clip['end_time']:.1f}s ({clip['duration']:.1f}s)")
            
            if clip['status'] == 'success':
                print(f"    S3 Key: {clip['s3_key']}")
            else:
                print(f"    Error: {clip.get('error', 'Unknown error')}")


def main():
    # Configuration
    ENDPOINT_URL = "YOUR_MODAL_ENDPOINT_HERE"  # Replace with your actual endpoint
    AUTH_TOKEN = "YOUR_AUTH_TOKEN_HERE"        # Replace with your actual token
    
    client = PodcastClipperClient(ENDPOINT_URL, AUTH_TOKEN)
    
    # Example 1: Process YouTube video
    print("Example 1: Processing YouTube Video")
    print("-" * 40)
    
    try:
        # Replace with an actual podcast YouTube URL
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        start_time = time.time()
        result = client.process_youtube_video(youtube_url, clip_count=2)
        end_time = time.time()
        
        client.print_results(result)
        print(f"\nProcessing time: {end_time - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"Error processing YouTube video: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Process S3 video
    print("Example 2: Processing S3 Video")
    print("-" * 40)
    
    try:
        # Replace with an actual S3 key
        s3_key = "your-folder/your-video.mp4"
        
        start_time = time.time()
        result = client.process_s3_video(s3_key, clip_count=1)
        end_time = time.time()
        
        client.print_results(result)
        print(f"\nProcessing time: {end_time - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"Error processing S3 video: {e}")


if __name__ == "__main__":
    main() 