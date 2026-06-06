import os
import re
import time
from datetime import datetime, date
from dotenv import load_dotenv
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()


def extract_video_id(url: str) -> str | None:
    patterns = [
        r'(?:v=|\/)([\w-]{11})(?:\?|&|$)',
        r'youtu\.be\/([\w-]{11})',
        r'(?:embed\/)([\w-]{11})',
        r'(?:shorts\/)([\w-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


class YouTubeScraper:
    def __init__(self, api_key: str | None = None):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_youtube_api_key_here":
            raise ValueError(
                "API Key YouTube tidak ditemukan. "
                "Setel GOOGLE_API_KEY di file .env"
            )
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def get_video_info(self, video_id: str) -> dict:
        request = self.youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )
        response = request.execute()
        if not response["items"]:
            raise ValueError(f"Video dengan ID '{video_id}' tidak ditemukan.")
        item = response["items"][0]
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        return {
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "published_at": snippet["publishedAt"],
            "comment_count": int(stats.get("commentCount", 0)),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
        }

    def scrape_comments(
        self,
        url: str,
        start_date: date | None = None,
        end_date: date | None = None,
        max_comments: int = 500,
        progress_callback=None,
    ) -> pd.DataFrame:
        video_id = extract_video_id(url)
        if not video_id:
            raise ValueError("URL YouTube tidak valid. Gunakan link video YouTube yang benar.")

        video_info = self.get_video_info(video_id)
        all_comments = []
        next_page_token = None
        fetched = 0
        page_number = 0

        while True:
            page_number += 1
            try:
                request = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText",
                )
                response = request.execute()
            except HttpError as e:
                if e.resp.status == 403:
                    reason = e.errors[0].get("reason", "") if e.errors else ""
                    if "commentsDisabled" in str(e):
                        raise ValueError("Komentar dinonaktifkan untuk video ini.")
                    elif "quotaExceeded" in reason or "quota" in str(e).lower():
                        raise ValueError("Kuota API Google habis. Tunggu hingga reset atau tambah kuota.")
                    else:
                        raise ValueError(f"Akses ditolak: {e}")
                elif e.resp.status == 404:
                    raise ValueError("Video tidak ditemukan atau telah dihapus.")
                else:
                    raise ValueError(f"Error API: {e}")

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comment_text = snippet["textDisplay"]
                published_at = snippet["publishedAt"]
                author = snippet["authorDisplayName"]
                like_count = snippet.get("likeCount", 0)
                published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))

                if start_date and published_dt.date() < start_date:
                    continue
                if end_date and published_dt.date() > end_date:
                    continue

                all_comments.append({
                    "video_id": video_id,
                    "video_title": video_info["title"],
                    "channel": video_info["channel"],
                    "author": author,
                    "comment": comment_text,
                    "published_at": published_at,
                    "like_count": like_count,
                })
                fetched += 1

                if max_comments > 0 and fetched >= max_comments:
                    break

            if progress_callback:
                progress_callback(fetched, max_comments, page_number)

            if max_comments > 0 and fetched >= max_comments:
                break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            time.sleep(0.1)

        if not all_comments:
            raise ValueError(
                "Tidak ada komentar ditemukan. "
                + ("Mungkin tidak ada komentar dalam rentang tanggal yang dipilih." if start_date or end_date else "")
            )

        df = pd.DataFrame(all_comments)
        return df
