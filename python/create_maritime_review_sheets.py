"""Build raw-image contact sheets for human review without drawing truth boxes."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def create_sheets(queue_path, output_dir):
    with Path(queue_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_session = defaultdict(list)
    for row in rows:
        by_session[int(row["session_index"])].append(row)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for session_index, session_rows in sorted(by_session.items()):
        profiles = list(dict.fromkeys(row["profile"] for row in session_rows))
        by_profile = {profile: sorted((row for row in session_rows
                            if row["profile"] == profile), key=lambda row: row["frame"])
                      for profile in profiles}
        columns = max(len(items) for items in by_profile.values())
        cell_width, image_height, caption_height = 640, 360, 28
        sheet = Image.new("RGB", (columns * cell_width,
                         len(profiles) * (image_height + caption_height)), "white")
        draw = ImageDraw.Draw(sheet)
        for profile_index, profile in enumerate(profiles):
            for frame_index, row in enumerate(by_profile[profile]):
                left = frame_index * cell_width
                top = profile_index * (image_height + caption_height)
                with Image.open(row["image_path"]) as raw:
                    thumb = raw.convert("RGB").resize((cell_width, image_height))
                sheet.paste(thumb, (left, top + caption_height))
                draw.text((left + 8, top + 6),
                          f"session {session_index} | {profile} | {row['frame']}",
                          fill="black")
        path = output_dir / f"session_{session_index}_raw_contact_sheet.jpg"
        if path.exists():
            raise FileExistsError(path)
        sheet.save(path, quality=92)
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    for path in create_sheets(args.queue, args.output_dir):
        print(path.resolve())
