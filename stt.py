import yt_dlp

url = "https://www.youtube.com/playlist?list=PLEneLIDJFpcA"

options = {
    "skip_download": True,
    "writesubtitles": True,
    "writeautomaticsub": True,
    "subtitleslangs": ["en.*"],
    "subtitlesformat": "srt",
    "outtmpl": "transcripts/%(playlist_index)03d - %(title)s.%(ext)s",

    # Reduce request pressure
    "sleep_interval": 5,
    "max_sleep_interval": 12,
    "sleep_interval_subtitles": 10,

    # Retry transient failures
    "retries": 5,
    "extractor_retries": 3,
    "ignoreerrors": True,

    # Use Deno for YouTube extraction
    "js_runtimes": {"deno": {}},
}

with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download([url])