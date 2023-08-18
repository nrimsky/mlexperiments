import os
import subprocess

def convert_mp4_to_grayscale_gif(mp4_filename, gif_filename):
    # First, generate a palette for the gif
    palette_cmd = [
        'ffmpeg',
        '-i', mp4_filename,
        '-vf', 'fps=2,scale=iw/2:ih/2:flags=lanczos,format=gray,palettegen',
        '-y', 'palette.png'
    ]
    subprocess.run(palette_cmd)
    
    # Convert the video to grayscale gif using the palette
    gif_cmd = [
        'ffmpeg',
        '-i', mp4_filename,
        '-i', 'palette.png',
        '-filter_complex', 
        'fps=2,scale=iw/2:ih/2:flags=lanczos,format=gray,setpts=0.5*PTS[x];[x][1:v]paletteuse',  # adjust the setpts value to control speed
        '-crf', '18',
        '-y', gif_filename
    ]
    subprocess.run(gif_cmd)

    # Clean up palette
    os.remove('palette.png')

if __name__ == '__main__':
    for file in os.listdir():
        if file.endswith(".mp4"):
            gif_name = file.rsplit('.', 1)[0] + '.gif'  # replace .mp4 with .gif
            convert_mp4_to_grayscale_gif(file, gif_name)
