import ffmpeg
import click
from pathlib import Path
from collections import defaultdict
from tempfile import TemporaryDirectory
from glob import glob


@click.group('shanghai_tech')
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


@main.command('train')
@click.option('--input-path', required=True)
@click.option('--output-path', required=True, type=click.Path(file_okay=True, resolve_path=True, writable=True))
@click.option('--frame-interval', default=16, type=int)
@click.option('--resize-shape', type=str)
def shanghai_tech_train(input_path, output_path, frame_interval, resize_shape):
    matches = glob(input_path)
    encode_args = {
        'c:v': 'libx264', 'crf': '18', 'pix_fmt': 'yuv420p', 'g': f'{frame_interval}',
        'profile:v': 'high', 'x264-params': f'keyint_min={frame_interval // 2}'}
    stream = merge_video_pipeline(list(matches), frame_interval)
    if resize_shape is not None:
        width, height = resize_shape.split(':')
        stream = stream.filter('scale', width=width, height=height)
    stream.output(output_path, **encode_args).run()


@main.command('test')
@click.option('--testing-dir', required=True)
@click.option('--output-dir', required=True, type=click.Path(file_okay=True, resolve_path=True, writable=True))
@click.option('--frame-interval', default=16, type=int)
@click.option('--resize-shape', type=str)
def shanghai_tech_test(testing_dir, output_dir, frame_interval, resize_shape):
    encode_args = {
        'c:v': 'libx264', 'crf': '18', 'pix_fmt': 'yuv420p', 'g': f'{frame_interval}',
        'profile:v': 'high', 'x264-params': f'keyint_min={frame_interval // 2}'}
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for sequence_dir in Path(testing_dir).glob('*'):
        stream = ffmpeg.input(str(sequence_dir / '*.jpg'), pattern_type='glob',
                              framerate=25)
        if resize_shape is not None:
            width, height = resize_shape.split(':')
            stream = stream.filter('scale', width=width, height=height)

        output_path = output_dir / f'{sequence_dir.name}.mp4'
        stream.output(str(output_path), **encode_args).run()


if __name__ == '__main__':
    main()
