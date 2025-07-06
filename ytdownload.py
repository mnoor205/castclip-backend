import yt_dlp

def download_video(url):
    ydl_opts = {
        # 'format': 'bestvideo[height=1080]+bestaudio/best[height=1080]/best',
        'format': 'bestvideo+bestaudio/bestaudio',
        'merge_output_format': 'mp4',
        'outtmpl': '%(title)s.%(ext)s',
        'noplaylist': True,
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=zAz2WtrdcMY"
    download_video(video_url)
