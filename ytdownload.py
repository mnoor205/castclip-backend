from pytubefix import YouTube
from pytubefix.cli import on_progress


url = "https://www.youtube.com/watch?v=sYj0exUT_Mw"
url2 = "https://www.youtube.com/watch?v=KUaacR-13gs"

yt = YouTube(url2, on_progress_callback=on_progress)
print(yt.title)

ys = yt.streams.get_highest_resolution()
ys.download()