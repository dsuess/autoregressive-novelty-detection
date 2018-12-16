FROM nvcr.io/nvidia/pytorch:18.11-py3

# Change working directory to PyTorch source path
WORKDIR /opt/

RUN rm -r pytorch && \
    git clone --recursive https://github.com/pytorch/pytorch

RUN cd pytorch && \
    git checkout v1.0.0 && \
    TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5+PTX" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    NCCL_INCLUDE_DIR="/usr/include/" \
    NCCL_LIB_DIR="/usr/lib/" \
    python setup.py install && python setup.py clean && \
    cd .. && \
    rm -r pytorch

ARG FFMPEG_VERSION=4.1

# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# minimal ffmpeg from source
RUN apt-get install -y yasm && \
    cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
      --prefix=/usr/local \
      --disable-static \
      --disable-all \
      --disable-autodetect \
      --disable-iconv \
      --enable-shared \
      --enable-avformat \
      --enable-avcodec \
      --enable-avfilter \
      --enable-protocol=file \
      --enable-demuxer=mov,matroska \
      --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb && \
    make -j8 && make install && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get remove -y yasm

# video_reader build deps
RUN pip install --upgrade cmake

# nvidia-docker only provides libraries for runtime use, not for
# development, to hack it so we can develop inside a container (not a
# normal or supported practice), we need to make an unversioned
# symlink so gcc can find the library.  Additional, different
# nvidia-docker versions put the lib in different places, so we make
# symlinks for both places.
RUN ln -s /usr/local/nvidia/lib64/libnvcuvid.so.1 /usr/local/lib/libnvcuvid.so && \
    ln -s libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so

ARG NVVL_COMMIT_ID=d7a306bfb9c349b6ab12365419551db53240e00e
ARG CUDA_HOME=/usr/local/cuda

RUN apt-get install -y unzip && \
    wget https://github.com/NVIDIA/nvvl/archive/$NVVL_COMMIT_ID.zip && \
    unzip $NVVL_COMMIT_ID.zip -d nvvl

RUN cd nvvl/nvvl-$NVVL_COMMIT_ID/pytorch1.0 && \
    python setup.py install && \
    apt-get remove -y unzip

# Reset default working directory
WORKDIR /workspace
