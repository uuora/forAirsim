"""Read DOCX body in document order; retain table cells and equation text."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

source, output = map(Path, sys.argv[1:3])
output.mkdir(parents=True, exist_ok=True)
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
def visible_text(node):
    if node.tag in (f"{{{ns['w']}}}del", f"{{{ns['w']}}}moveFrom"):
        return ""
    if node.tag in (f"{{{ns['w']}}}t", f"{{{ns['m']}}}t"):
        return node.text or ""
    if node.tag in (f"{{{ns['w']}}}br", f"{{{ns['w']}}}tab"):
        return " "
    return "".join(visible_text(child) for child in node)

with zipfile.ZipFile(source) as archive:
    root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", ns)
    rows = []
    for index, block in enumerate(body, 1):
        if block.tag.endswith("}p"):
            text = visible_text(block)
            rows.append({"body_index": index, "kind": "paragraph", "text": text})
        elif block.tag.endswith("}tbl"):
            cells = [["\n".join(visible_text(p) for p in cell.findall("w:p", ns))
                      for cell in row.findall("w:tc", ns)] for row in block.findall("w:tr", ns)]
            rows.append({"body_index": index, "kind": "table", "cells": cells})
    media = [name for name in archive.namelist() if name.startswith("word/media/")]
    image_folder = output / "media"
    image_folder.mkdir(exist_ok=True)
    for name in media:
        if Path(name).suffix.lower() in (".png", ".jpg", ".jpeg"):
            (image_folder / Path(name).name).write_bytes(archive.read(name))
metadata = {"source": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "blocks": rows, "embedded_media": media,
            "scope": "Final text view includes insertions and excludes deletions/field instructions; image text not OCRed; formatting not rendered"}
(output / "proposal_extract.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
lines = []
for row in rows:
    lines.append(f"[body {row['body_index']} {row['kind']}]")
    lines.append(row["text"] if row["kind"] == "paragraph" else "\n".join(" | ".join(c) for c in row["cells"]))
(output / "proposal_extract.txt").write_text("\n\n".join(lines), encoding="utf-8")
print(json.dumps({"blocks": len(rows), "media": len(media), "sha256": metadata["sha256"]}))
