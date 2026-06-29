from pathlib import Path
import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
root = Path('datasets/piper_insert_mouse_battery_lerobot/videos/chunk-000')
out_dir = root / 'observation.images.cam_vertical'
out_dir.mkdir(parents=True, exist_ok=True)
made = 0
for top in sorted((root / 'observation.images.top_head').glob('episode_*.mp4')):
    name = top.name
    out = out_dir / name
    if out.exists() and out.stat().st_size > 0:
        continue
    left = root / 'observation.images.hand_left' / name
    right = root / 'observation.images.hand_right' / name
    if not left.exists() or not right.exists():
        raise FileNotFoundError(name)
    cmd = [
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
        '-i', str(top), '-i', str(left), '-i', str(right),
        '-filter_complex', '[0:v][1:v][2:v]vstack=inputs=3[v]', '-map', '[v]',
        '-r', '30', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', str(out),
    ]
    subprocess.run(cmd, check=True)
    made += 1
    if made % 10 == 0:
        print(f'generated {made}', flush=True)
print(f'generated_new={made}', flush=True)
print(f'total={len(list(out_dir.glob("episode_*.mp4")))}', flush=True)
