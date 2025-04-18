# HoloCrack

A C# application for crack detection and analysis using deep learning and computer vision techniques, designed for integration with HoloLens 2 applications. The name "HoloCrack" reflects its intended use with Microsoft HoloLens 2 mixed reality platform.

## Features

- Deep learning-based crack segmentation using ONNX runtime
- Crack measurement analysis (width, length, area)
- Visualization of detected cracks with measurements
- Support for various image formats

## Requirements

- .NET Core 3.1 or later
- OpenCVSharp4
- ONNX Runtime
- Visual Studio 2019 or later

## Project Structure

- `Core/` - Core functionality implementation
  - `ImagePreprocessing.cs` - Image preprocessing utilities
  - `PostProcessing.cs` - Post-processing and analysis tools
- `Program.cs` - Main application entry point
- `Models/` - Contains the ONNX model file
- `Data/` - Sample images for testing

## Usage

1. Place your crack images in the `Data` folder
2. Place your trained model in the `Models` folder as `segmentation_model.onnx`
3. Run the application
4. Check the output files:
   - `raw_mask.png` - Raw segmentation mask
   - `cleaned_mask.png` - Processed mask
   - `mask_overlay_transparent.png` - Overlay visualization
   - `crack_analysis_overlay.png` - Analysis visualization with measurements

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. 