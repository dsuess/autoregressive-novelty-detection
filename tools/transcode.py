import ffmpeg
import click
from pathlib import Path
from collections import defaultdict
from tempfile import TemporaryDirectory
from glob import glob


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
@click.option('--input-path', required=True)
@click.option('--output-path', required=True, type=click.Path(file_okay=True, resolve_path=True, writable=True))
@click.option('--frame-interval', default=16, type=int)
def shanghai_tech(input_path, output_path, frame_interval):
    matches = glob(input_path)
    encode_args = {
        'c:v': 'libx264', 'crf': '18', 'pix_fmt': 'yuv420p', 'g': f'{frame_interval}',
        'profile:v': 'high', 'x264-params': f'keyint_min={frame_interval // 2}'}
    stream = merge_video_pipeline(list(matches), frame_interval)
    stream.output(output_path, **encode_args).run()


if __name__ == '__main__':
    main()
