# RetroShade

**RetroShade** is a lightweight, GPU-accelerated image and video editing library featuring a Qt-based user interface. It leverages a custom build of OpenCV 4.10.0 for efficient media I/O and CUDA 12.6 for high-performance processing. Future plans include OpenCL support to extend compatibility to non-NVIDIA GPUs.

## ✨ Features

- **Qt 6.8.2 UI**: Intuitive and responsive user interface built with Qt.
- **Custom OpenCV 4.10.0**: Streamlined OpenCV build optimized for image and video I/O.
- **CUDA 12.6 Acceleration**: GPU-powered effects for real-time processing.
- **OpenCL Acceleration**: All effects are also implemented using OpenCL for broader GPU compatibility.
- **Modular Effect System**: Easily extendable architecture for adding new effects.
- **Sample Projects**: Includes example projects demonstrating various effects.

## 📁 Project Structure

```
VideoEditorBeta/
│
├── bin/                          # Precompiled binaries (e.g., DLLs)
│   └── *.dll
│
├── lib/                          # Static or dynamic libraries (if any)
│
├── include/                      # External include files (if any)
│
├── Header Files/                # General header files
│   ├── utils.h
│   ├── timer.h
│   ├── image.h
│   └── ...
│
├── Source Files/                # Main application source code
│   ├── main.cpp
│   ├── utils.cpp
│   ├── image.cpp
│   └── ...
│
├── Resource Files/              # Resource-related files
│   ├── resource.h
│   ├── VideoEditorBeta.rc
│   └── Resource.qrc
│
├── UI/                          # User interface components
│   ├── mainWindow.cpp/.h/.ui
│   ├── EffectButton.cpp/.h
│   └── ColorButton.cpp/.h
│
└── Effects/                     # Video/image processing effects
    ├── <EffectName>/            # One folder per effect
    │   ├── Image/               # Image-specific effect implementation
    │   │   ├── image<EffectName>.h
    │   │   └── image<EffectName>.cpp
    │   │
    │   ├── Video/               # Video-specific effect implementation
    │   │   ├── video<EffectName>.h
    │   │   └── video<EffectName>.cpp
    │   │
    │   └── Launcher/            # Shared launcher logic for the effect
    │       ├── <EffectName>.h
    │       ├── <EffectName>.cpp
    │       └── CUDA/            # CUDA-specific GPU code
    │           ├── <EffectName>_launcher.cu
    │           └── <EffectName>_launcher.cuh
    │
    └── ...
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

- **✅ OpenCL Support**: Complete support added for AMD and Intel GPUs.
- **Additional Effects**: Implement more video effects such as sharpening, color correction, and transitions.
- **Cross-Platform Support**: Enhance compatibility with macOS and Linux systems.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📫 Contact

For questions or suggestions, please open an issue on the [GitHub repository](https://github.com/erfan-ops/VideoEditorBeta/issues).

---
