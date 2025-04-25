# VideoEditorBeta

**VideoEditorBeta** is a lightweight, GPU-accelerated image and video editing library featuring a Qt-based user interface. It leverages a custom build of OpenCV 4.10.0 for efficient media I/O and CUDA 12.6 for high-performance processing. Future plans include OpenCL support to extend compatibility to non-NVIDIA GPUs.

## ✨ Features

- **Qt 6.8.2 UI**: Intuitive and responsive user interface built with Qt.
- **Custom OpenCV 4.10.0**: Streamlined OpenCV build optimized for image and video I/O.
- **CUDA 12.6 Acceleration**: GPU-powered effects for real-time processing.
- **Modular Effect System**: Easily extendable architecture for adding new effects.
- **Sample Projects**: Includes example projects demonstrating various effects.

## 🛠️ Effects Implemented

- **Black & White**: Convert videos to grayscale using CUDA.
- **Blur**: Apply Gaussian blur effects with GPU acceleration.
- **Censor**: Obscure sensitive areas in videos using CUDA-based techniques.

## 📁 Project Structure

```
VideoEditorBeta/
├── include/                 # Header files
├── lib/                     # Compiled libraries
├── samples/                 # Sample projects demonstrating effects
├── src/                     # Source files
│   ├── EffectButton.cpp/.h  # UI components for effect buttons
│   ├── ImageOutlines.h      # Image outline definitions
│   ├── Video.cpp/.h         # Core video processing classes
│   ├── effects/             # CUDA implementations of effects
│   │   ├── blackAndWhite_launcher.cu/.cuh
│   │   ├── blur_launcher.cu/.cuh
│   │   ├── censor_launcher.cu/.cuh
├── resources/               # Resource files (e.g., icons, UI layouts)
│   ├── MainWindow.ui        # Main window layout
│   ├── Resource.qrc         # Qt resource file
├── VideoEditorBeta.sln      # Visual Studio solution file
├── VideoEditorBeta.vcxproj  # Visual Studio project file
├── README.md                # Project documentation
├── LICENSE                  # MIT License
```

## 🛠️ How to Build (Visual Studio)

### Prerequisites

- [**Visual Studio 2022**](https://visualstudio.microsoft.com/vs/)
  - With **Desktop development with C++** workload
- [**Qt 6.8.2**](https://www.qt.io/download)
- [**CUDA Toolkit 12.6**](https://developer.nvidia.com/cuda-downloads)
- [**OpenCV 4.10.0 (custom build)**] – already included

### Steps

1. **Clone the Repository**  
   Open a terminal or Git Bash and run:
   ```bash
   git clone https://github.com/erfan-ops/VideoEditorBeta.git
   cd VideoEditorBeta
   ```

2. **Open the Solution**  
   - Launch **Visual Studio**.
   - Open `VideoEditorBeta.sln`.

3. **Configure Project**  
   - Ensure `Qt` and `CUDA` include/lib paths are correctly set in **Project Properties**.
   - You may need to adjust:
     - `VC++ Directories` → Include and Library paths
     - `Linker` → Input (e.g., CUDA libraries)

4. **Build the Project**  
   - Set build mode to `Release` or `Debug`.
   - Press **Ctrl + Shift + B** or select **Build → Build Solution**.

5. **Run the App**  
   - After building, run the executable from `x64/Release/` or directly from Visual Studio.

## 📌 Future Plans

- **OpenCL Support**: Extend GPU acceleration to AMD and Intel GPUs.
- **Additional Effects**: Implement more video effects such as sharpening, color correction, and transitions.
- **Cross-Platform Support**: Enhance compatibility with macOS and Linux systems.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📫 Contact

For questions or suggestions, please open an issue on the [GitHub repository](https://github.com/erfan-ops/VideoEditorBeta/issues).

---
