# VideoEditorBeta

**VideoEditorBeta** is a high-performance video editing tool that provides a wide range of video effects. The project leverages CUDA for fast processing and OpenCV for image manipulation, enabling real-time video editing and effect rendering.

## Features

- **CUDA-Accelerated Effects**: Various video effects are processed using CUDA, improving performance by utilizing GPU power.
- **OpenCV Integration**: OpenCV is used for efficient video frame manipulation, including filtering and transformations.
- **Support for Multiple Effects**: Apply effects like pixelation, color filtering, and custom transformations to enhance video content.
- **Audio Extraction and Merge**: Extract audio from the input video and merge it with the processed video.

## Requirements

- **CUDA**: Version 11 or later for GPU-accelerated performance.
- **OpenCV**: Version 4.5 or later for video processing.
- **FFmpeg**: For audio extraction and final video merging.
- **C++17**: For compiling the project.

## Compilation

To compile the project, make sure you have the required dependencies installed (CUDA, OpenCV, and FFmpeg), then use the following commands:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

To use the `videoVintage8bit` effect, call the function with the required parameters:

```cpp
videoVintage8bit(
    L"input_video.mp4",  // Path to the input video
    L"output_video.mp4", // Path to the output video
    8,                  // Pixel width for pixelation
    8,                  // Pixel height for pixelation
    color1,              // Color 1 in BGR format (unsigned char[3])
    color2,              // Color 2 in BGR format (unsigned char[3])
    color3,              // Color 3 in BGR format (unsigned char[3])
    64,           // Threshold for color rounding
    8,                   // Line width for line effects
    10                   // Line darkening threshold
);
```

### Parameters

- `inputPath`: Path to the input video.
- `outputPath`: Path to the output video.
- `pixelWidth`, `pixelHeight`: Dimensions for pixelation effect.
- `color1`, `color2`, `color3`: Colors to apply in the vintage effect.
- `threshold`: Threshold value for color rounding.
- `lineWidth`: Width of horizontal lines for the line effect.
- `lineDarkeningThresh`: Threshold for line darkening.

### Example

```cpp
unsigned char color1[3] = {255, 0, 0}; // Blue
unsigned char color2[3] = {0, 255, 0}; // Green
unsigned char color3[3] = {0, 0, 255}; // Red

videoVintage8bit(
    L"example_input.mp4",
    L"example_output.mp4",
    8, 8, color1, color2, color3, 64, 8, 10
);
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [CUDA](https://developer.nvidia.com/cuda-zone) for GPU-accelerated computing.
- [OpenCV](https://opencv.org/) for computer vision and image processing.
- [FFmpeg](https://ffmpeg.org/) for audio extraction and video merging.
