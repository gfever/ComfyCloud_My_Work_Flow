import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

print('Quick smoke generator starting...')
WORKSPACE = os.getcwd()
SMOKE_WIDTH = 256
SMOKE_HEIGHT = 256
SMOKE_FRAMES = 8
SMOKE_FPS = 8
SMOKE_FRAMES_DIR = os.path.join(WORKSPACE, 'smoke_frames')
os.makedirs(SMOKE_FRAMES_DIR, exist_ok=True)

prompt_text = 'Test generation — quick smoke'
text_preview = (prompt_text or '')[:120]

for i in range(SMOKE_FRAMES):
    r = (30 + i * 20) % 256
    g = (80 + i * 10) % 256
    b = (160 + i * 5) % 256
    img = Image.new('RGB', (SMOKE_WIDTH, SMOKE_HEIGHT), (r, g, b))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if font:
        try:
            w, h = font.getsize(text_preview)
        except Exception:
            bbox = draw.textbbox((0,0), text_preview, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
    else:
        try:
            bbox = draw.textbbox((0,0), text_preview)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = 100, 20
    x = int((SMOKE_WIDTH - w) * i / max(1, SMOKE_FRAMES - 1))
    y = SMOKE_HEIGHT // 2 - h // 2
    draw.text((x, y), text_preview, fill=(255, 255, 255), font=font)
    cx = int(SMOKE_WIDTH * (0.2 + 0.6 * (i / max(1, SMOKE_FRAMES - 1))))
    cy = int(SMOKE_HEIGHT * 0.25)
    r0 = 12
    draw.ellipse((cx - r0, cy - r0, cx + r0, cy + r0), fill=(255, 200, 0))
    draw.text((6, SMOKE_HEIGHT - 14), f'frame {i+1}/{SMOKE_FRAMES}', fill=(230,230,230), font=font)
    path = os.path.join(SMOKE_FRAMES_DIR, f"{i}.png")
    img.save(path)
    print('Saved', path)

smoke_video = os.path.join(WORKSPACE, 'SMOKE_OUTPUT.mp4')
ffmpeg_cmd = [
    'ffmpeg', '-y', '-framerate', str(SMOKE_FPS), '-i', os.path.join(SMOKE_FRAMES_DIR, '%d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', smoke_video
]
print('Running ffmpeg to assemble video...')
try:
    subprocess.check_call(ffmpeg_cmd)
    print('Smoke video saved:', smoke_video)
except Exception as e:
    print('ffmpeg failed or not available:', e)
    print('You can assemble frames with:')
    print("ffmpeg -framerate {} -i {}/%d.png -c:v libx264 -pix_fmt yuv420p -preset fast {}".format(SMOKE_FPS, SMOKE_FRAMES_DIR, smoke_video))

print('Done.')
