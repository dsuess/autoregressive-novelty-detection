import ffmpeg
import click
from pathlib import Path
from collections import defaultdict
from tempfile import TemporaryDirectory


@click.group('main')
def main():
    pass


def single_video_pipeline(path, frame_interval):
    frame_number = int(ffmpeg.probe(path)['streams'][0]['nb_frames'])
    end_frame = (frame_number // frame_interval) * frame_interval - 1
    stream = ffmpeg.input(path).trim(end_frame=end_frame)
    stream = ffmpeg.filter(stream, 'setsar', sar='1/1')
    stream = ffmpeg.filter(stream, 'setdar', dar='16/9')
    return stream


def merge_video_pipeline(paths, frame_interval):
    single_pipelines = [single_video_pipeline(p, frame_interval)
                        for p in sorted(paths)]
    return ffmpeg.concat(*single_pipelines)




@main.command('shanghai-tech')
@click.option('--input-dir', required=True, type=click.Path(file_okay=False, resolve_path=True, writable=False))
@click.option('--output-dir', required=True, type=click.Path(file_okay=False, resolve_path=True, writable=True))
@click.option('--frame-interval', default=16, type=int)
def shanghai_tech(input_dir, output_dir, frame_interval):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    matches = input_dir.glob('*.avi')
    grouped_matches = defaultdict(list)
    for match in matches:
        key = str(match.stem).split('_')[0]
        grouped_matches[key].append(str(match))
    output_dir.mkdir(exist_ok=True)
    encode_args = {
        'c:v': 'libx264', 'crf': '18', 'pix_fmt': 'yuv420p',
        'x264-params': f'keyint={frame_interval}',
        'profile:v': 'high',
    }

    for key, paths in grouped_matches.items():
        merged_pipeline = merge_video_pipeline(paths, frame_interval)
        merged_pipeline \
            .output(str(output_dir / f'{key}.mp4'), **encode_args) \
            .run()


if __name__ == '__main__':
    main()
