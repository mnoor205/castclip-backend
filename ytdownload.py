import yt_dlp

def download_video(url):
    ydl_opts = {
        'format': 'bestvideo[height=1440]+bestaudio/best[height=1440]/best',
        'merge_output_format': 'mp4',
        'outtmpl': '%(title)s.%(ext)s',
        'noplaylist': True,
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=-UzJOk85OZI"
    download_video(video_url)
