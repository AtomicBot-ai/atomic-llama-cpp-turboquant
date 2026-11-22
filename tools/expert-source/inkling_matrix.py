"""Render the Inkling Metal benchmark's measured default-profile matrix."""

import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=None)
def font(size, bold=False):
    filename = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(str(Path("/System/Library/Fonts/Supplemental") / filename), size)
    except OSError:
        return ImageFont.load_default(size=size)


def host_label():
    def sysctl(key):
        result = subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    chip = sysctl("machdep.cpu.brand_string")
    memory = sysctl("hw.memsize")
    parts = [chip] if chip else []
    if memory.isdigit():
        parts.append(f"{int(memory) / 2**30:g} GB unified memory")
    return "  ·  ".join(parts)


def write_matrix(directory, results, quants, budgets, defaults, depth, draft_name):
    rows = {(row["budget_gib"], row["quant"]): row for row in results["rows"]}
    profiles = [f"DSpark {depth or defaults[q]}/32" for q in quants]
    hardware = host_label()
    prompt = results["prompt"]
    details = f"{prompt}  ·  4,000 input tokens  ·  64 output tokens"
    if hardware:
        details = hardware + "  ·  " + details

    print(f"\nInkling 2 Mini    {details}")
    print(" " * 22 + "".join(f"{q + ' · ' + profile:^25}" for q, profile in zip(quants, profiles)))
    print(f"{'Engine budget':<22}" + f"{'TTFT (s)':>11}{'Decode (tok/s)':>14}" * len(quants))

    margin, budget_width, group_width = 42, 260, 320
    width = 2 * margin + budget_width + group_width * len(quants)
    top, header_height, row_height = 210, 108, 52
    bottom = top + header_height + row_height * (len(budgets) + 1) + 12
    image = Image.new("RGB", (width, bottom + 91), "#f5f5f0")
    draw = ImageDraw.Draw(image)
    ink, green = "#132b27", "#075c47"

    def label(x, y, value, size, color=ink, bold=False, anchor="lm"):
        draw.text((x, y), value, font=font(size, bold), fill=color, anchor=anchor)

    label(margin, 33, "ATOMIC ENGINE   /   METAL", 19, green, True)
    label(margin, 82, "Inkling 2 Mini inference", 48, ink, True)
    label(margin, 130, details, 22)
    label(margin, 162, "GGUF experts via pread  ·  1,024-token prefill blocks  ·  Q4_0 KV cache", 19)
    label(margin, 187, f"Draft: {draft_name}", 16)

    draw.rounded_rectangle((margin, top, width-margin, top+header_height), 12, fill="#193c36")
    label(margin+15, top+55, "Engine budget", 22, "white", True)
    for col, (quant, profile) in enumerate(zip(quants, profiles)):
        left = margin + budget_width + group_width * col
        center = left + group_width // 2
        draw.line((left, top+10, left, top+header_height-6), fill="#58716b")
        label(center, top+26, quant, 26, "white", True, "mm")
        label(center, top+55, profile, 19, "#dcebe6", anchor="mm")
        label(left+80, top+84, "TTFT (s) ↓", 18, "white", True, "mm")
        label(left+240, top+84, "Decode (tok/s) ↑", 18, "white", True, "mm")

    for i, budget in enumerate([*budgets, None]):
        stock = budget is None
        y = top + header_height + i * row_height + (12 if stock else 0)
        title = "Stock llama.cpp" if stock else f"{budget} GiB + KV"
        fill = "#dceaf0" if stock else ("#ffffff", "#e9efe9")[i % 2]
        draw.rectangle((margin, y, width-margin, y+row_height-1), fill=fill)
        label(margin+15, y+(17 if stock else 26), title, 22, ink, True)
        if stock:
            label(margin+15, y+38, "Uncapped", 16)
        line = f"{title:<22}"
        for col, quant in enumerate(quants):
            left = margin + budget_width + group_width * col
            row = rows[budget, quant]
            ttft, tps = row["ttft_s"], row["decode_tps"]
            line += f"{ttft:11.2f}{tps:14.2f}"
            draw.line((left, y, left, y+row_height-1), fill="#cedbd5")
            label(left+80, y+26, f"{ttft:.2f}", 23, ink, anchor="mm")
            label(left+240, y+26, f"{tps:.2f}", 23, green if not stock else ink, True, "mm")
        print(line)
    label(margin, bottom+28, "Each Atomic budget adds 100 MiB for KV. TTFT starts at request submission with the model loaded.", 18)
    label(margin, bottom+56, "One request per cell pair. All 64 output IDs matched fresh stock for each GGUF.", 18)
    path = directory / "matrix.png"
    image.save(path)
    print(f"Matrix image: {path}", flush=True)
