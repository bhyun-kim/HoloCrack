using System;
using System.IO;
using System.Linq;
using HoloCrack.Core;           
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace HoloCrack
{
    class Program
    {
        static void Main()
        {
            // 1. Load model
            using var session = new InferenceSession("Models/segmentation_model.onnx");
            var inputName = session.InputMetadata.Keys.First();
            var outputName = session.OutputMetadata.Keys.First();

            // 2. Load Image
            string originalImagePath = "Data/sample_crack.jpg";
            Mat originalBgr = Cv2.ImRead(originalImagePath, ImreadModes.Color);
            if (originalBgr.Empty())
            {
                Console.WriteLine($"Could not load original image: {originalImagePath}");
                return;
            }

            // 3. Apply Preprocessing
            DenseTensor<float> inputTensor = ImagePreprocessing.LoadAndPrepareImage(
                imagePath: originalImagePath,
                inputWidth: 1024,
                inputHeight: 1024
            );

            // 4. Run Inference
            var inputValue = NamedOnnxValue.CreateFromTensor(inputName, inputTensor);
            using var results = session.Run(new[] { inputValue });
            var outputTensor = results.First(r => r.Name == outputName).AsTensor<float>();

            // 5. Create a mask that matches with original image size
            Mat upMaskGray = CreateMaskFromOutput(outputTensor);
            // Resize to original image size
            Mat resizedMask = new Mat();
            Cv2.Resize(upMaskGray, resizedMask, new Size(originalBgr.Cols, originalBgr.Rows), 0, 0, InterpolationFlags.Linear);
            upMaskGray = resizedMask;

            // Save the raw mask
            Cv2.ImWrite("raw_mask.png", upMaskGray);

            // Add preprocessing steps before crack analysis
            Mat cleanedMask = PreprocessMaskForAnalysis(upMaskGray);
            // Save the cleaned mask
            Cv2.ImWrite("cleaned_mask.png", cleanedMask);

            // 6. Run crack Analysis with higher minimum pixels
            var measurements = PostProcessing.AnalyzeCracks(
                cleanedMask, 
                new Size(originalBgr.Cols, originalBgr.Rows), 
                minPixels: 500
            );
            PrintMeasurements(measurements);

            // 7. Visualize Overlay crack segmentation results
            Mat blendedOverlay = CreateOverlay(originalBgr, upMaskGray);
            Cv2.ImWrite("mask_overlay_transparent.png", blendedOverlay);

            // 8. Visualize crack analyze result
            Mat analysisVisualization = PostProcessing.VisualizeAnalysis(originalBgr, measurements);
            Cv2.ImWrite("crack_analysis_overlay.png", analysisVisualization);
        }

        private static Mat CreateMaskFromOutput(Tensor<float> outputTensor)
        {
            var dims = outputTensor.Dimensions;
            int channels = dims[1];
            int outHeight = dims[2];
            int outWidth = dims[3];
            float[] outputArray = outputTensor.ToArray();

            // Calculate x8 dimensions (model's required upscaling)
            int modelUpscaledHeight = outHeight * 8;
            int modelUpscaledWidth = outWidth * 8;

            if (channels > 1)
            {
                float[] softmaxOutput = PostProcessing.ApplySoftmax(outputArray, channels, outHeight, outWidth);
                Mat probMask = PostProcessing.CreateProbabilityMask(softmaxOutput, outHeight, outWidth);
                
                // Upscale x8 as part of model inference
                Mat upscaledMask = new Mat();
                Cv2.Resize(probMask, upscaledMask, new Size(modelUpscaledWidth, modelUpscaledHeight), 0, 0, InterpolationFlags.Linear);
                
                return PostProcessing.ProcessMask(upscaledMask);
            }
            else
            {
                Mat rawMask = new Mat(outHeight, outWidth, MatType.CV_8UC1);
                for (int i = 0; i < outHeight * outWidth; i++)
                {
                    float val = Math.Clamp(outputArray[i], 0, 1);
                    byte gray = (byte)(val * 255f);
                    rawMask.Set(i / outWidth, i % outWidth, gray);
                }
                
                // Upscale x8 as part of model inference
                Mat upscaledMask = new Mat();
                Cv2.Resize(rawMask, upscaledMask, new Size(modelUpscaledWidth, modelUpscaledHeight), 0, 0, InterpolationFlags.Linear);
                return upscaledMask;
            }
        }

        private static Mat CreateOverlay(Mat originalBgr, Mat mask)
        {
            // Resize mask to match original image size

            Mat redOverlay = new Mat(originalBgr.Size(), originalBgr.Type(), new Scalar(0, 0, 0));
            for (int y = 0; y < originalBgr.Rows; y++)
            {
                for (int x = 0; x < originalBgr.Cols; x++)
                {
                    if (mask.At<byte>(y, x) > 127)
                    {
                        redOverlay.Set(y, x, new Vec3b(0, 0, 255));
                    }
                }
            }

            Mat blendedOverlay = new Mat();
            Cv2.AddWeighted(originalBgr, 1.0, redOverlay, 0.5, 0.0, blendedOverlay);
            return blendedOverlay;
        }

        private static void PrintMeasurements(List<CrackMeasurement> measurements)
        {
            Console.WriteLine("\nCrack Measurements:");
            foreach (var (crack, index) in measurements.Select((c, i) => (c, i)))
            {
                Console.WriteLine($"Crack {index + 1}:");
                Console.WriteLine($"  Width: {crack.Width:F1} pixels");
                Console.WriteLine($"  Length: {crack.Length:F1} pixels");
                Console.WriteLine($"  Center: ({crack.CenterX:F1}, {crack.CenterY:F1})");
            }
        }

        private static Mat PreprocessMaskForAnalysis(Mat mask)
        {
            Mat cleanedMask = new Mat();
            
            // Apply binary threshold to make the mask more decisive
            Cv2.Threshold(mask, cleanedMask, 127, 255, ThresholdTypes.Binary);
            
            // Create kernel for morphological operations
            var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(3, 3));
            
            // Remove small noise
            Cv2.MorphologyEx(cleanedMask, cleanedMask, MorphTypes.Open, kernel);
            
            // Connect nearby components
            Cv2.MorphologyEx(cleanedMask, cleanedMask, MorphTypes.Close, kernel);
            
            return cleanedMask;
        }
    }
}
