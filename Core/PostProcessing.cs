using System;
using OpenCvSharp;
using System.Collections.Generic;
using System.Linq;

namespace HoloCrack.Core
{
    public static class PostProcessing
    {
        public static float[] ApplySoftmax(float[] outputArray, int channels, int height, int width)
        {
            float[] softmaxOutput = new float[outputArray.Length];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Find max value for numerical stability
                    float maxVal = float.MinValue;
                    for (int c = 0; c < channels; c++)
                    {
                        int idx = c * height * width + y * width + x;
                        if (outputArray[idx] > maxVal)
                            maxVal = outputArray[idx];
                    }

                    // Compute sum of exponentials
                    float sumExp = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        int idx = c * height * width + y * width + x;
                        float expVal = (float)Math.Exp(outputArray[idx] - maxVal);
                        softmaxOutput[idx] = expVal;
                        sumExp += expVal;
                    }

                    // Normalize to get probabilities
                    for (int c = 0; c < channels; c++)
                    {
                        int idx = c * height * width + y * width + x;
                        softmaxOutput[idx] /= sumExp;
                    }
                }
            }

            return softmaxOutput;
        }

        public static Mat CreateProbabilityMask(float[] softmaxOutput, int height, int width)
        {
            Mat probMask = new Mat(height, width, MatType.CV_32F);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = 1 * height * width + y * width + x;
                    float prob = softmaxOutput[idx];
                    probMask.Set(y, x, prob);
                }
            }
            return probMask;
        }

        public static Mat ProcessMask(Mat probMask)
        {
            // Convert to 8-bit grayscale
            Mat maskGray = new Mat();
            probMask.ConvertTo(maskGray, MatType.CV_8UC1, 255.0);
            return maskGray;
        }

        public static List<CrackMeasurement> AnalyzeCracks(Mat binaryMask, Size originalSize, int minPixels = 100)
        {
            // First resize the mask to match original image size if needed
            Mat resizedMask = new Mat();
            if (binaryMask.Size() != originalSize)
            {
                Cv2.Resize(binaryMask, resizedMask, originalSize, 0, 0, InterpolationFlags.Linear);
            }
            else
            {
                resizedMask = binaryMask.Clone();
            }

            Mat labels = new Mat();
            Mat stats = new Mat();
            Mat centroids = new Mat();
            int numLabels = Cv2.ConnectedComponentsWithStats(resizedMask, labels, stats, centroids);

            List<CrackMeasurement> measurements = new List<CrackMeasurement>();

            // Process each crack (skip label 0 which is background)
            for (int label = 1; label < numLabels; label++)
            {
                // Get area from stats matrix (CC_STAT_AREA is the 4th column)
                int area = stats.At<int>(label, 4); // Get area of the component
                
                // Skip if area is less than minimum pixels
                if (area < minPixels)
                    continue;

                // Extract single crack
                Mat crackMask = new Mat();
                Cv2.Compare(labels, label, crackMask, CmpType.EQ);
                
                // Find skeleton using distance transform and local maxima
                Mat dist = new Mat();
                Cv2.DistanceTransform(crackMask, dist, DistanceTypes.L2, DistanceTransformMasks.Mask3);
                
                // Get medial axis points
                Mat localMax = new Mat();
                Cv2.Dilate(dist, localMax, null);
                Mat medialAxis = new Mat();
                Cv2.Compare(dist, localMax, medialAxis, CmpType.EQ);
                
                // Add this line to mask the medial axis with the original crack mask
                Cv2.BitwiseAnd(medialAxis, crackMask, medialAxis);
                
                // Calculate measurements
                double minVal, maxVal;
                Point minLoc, maxLoc;
                Cv2.MinMaxLoc(dist, out minVal, out maxVal, out minLoc, out maxLoc);
                double maxDist = maxVal; // Maximum width/2
                double length = Cv2.CountNonZero(medialAxis) * 0.8; // Approximate length
                
                measurements.Add(new CrackMeasurement(maxDist * 2, length, centroids.At<double>(label, 0), centroids.At<double>(label, 1), medialAxis.Clone(), area));
            }

            return measurements;
        }

        public static Mat VisualizeAnalysis(Mat originalImage, List<CrackMeasurement> measurements)
        {
            // Create a copy of the original image for overlay
            Mat visualResult = originalImage.Clone();
            
            foreach (var (crack, index) in measurements.Select((c, i) => (c, i)))
            {
                // Generate a random color for this crack
                Random rng = new Random(index);
                Vec3b color = new Vec3b(
                    0,                          // B
                    (byte)rng.Next(128, 256),   // G
                    (byte)rng.Next(128, 256)    // R
                );
                
                // Directly color the skeleton pixels
                for (int y = 0; y < originalImage.Rows; y++)
                {
                    for (int x = 0; x < originalImage.Cols; x++)
                    {
                        if (crack.MedialAxis.At<byte>(y, x) > 127)
                        {
                            visualResult.Set(y, x, color);
                        }
                    }
                }
                
                // Draw measurements text
                Point textPos = new Point(
                    crack.CenterX,
                    crack.CenterY
                );

                // Function to draw outlined text
                Action<string, Point> drawOutlinedText = (text, pos) => {
                    // Draw black outline
                    foreach (var offset in new[] {
                        (-1,-1), (-1,1), (1,-1), (1,1),
                        (-1,0), (1,0), (0,-1), (0,1)
                    })
                    {
                        Cv2.PutText(visualResult, 
                            text,
                            new Point(pos.X + offset.Item1, pos.Y + offset.Item2),
                            HersheyFonts.HersheySimplex,
                            0.5,
                            new Scalar(0, 0, 0),
                            2,
                            LineTypes.AntiAlias);
                    }
                    // Draw white text
                    Cv2.PutText(visualResult,
                        text,
                        pos,
                        HersheyFonts.HersheySimplex,
                        0.5,
                        new Scalar(255, 255, 255),
                        1,
                        LineTypes.AntiAlias);
                };

                // Draw text with outline
                double lineHeight = 20;
                drawOutlinedText($"Crack {index + 1}", textPos);
                drawOutlinedText($"W:{crack.Width:F1}px", new Point(textPos.X, textPos.Y + lineHeight));
                drawOutlinedText($"L:{crack.Length:F1}px", new Point(textPos.X, textPos.Y + lineHeight * 2));
                drawOutlinedText($"A:{crack.Area}px", new Point(textPos.X, textPos.Y + lineHeight * 3));
            }

            return visualResult;
        }
    }

    public class CrackMeasurement
    {
        public CrackMeasurement(double width, double length, double centerX, double centerY, Mat medialAxis, int area)
        {
            Width = width;
            Length = length;
            CenterX = centerX;
            CenterY = centerY;
            MedialAxis = medialAxis;
            Area = area;
        }

        public double Width { get; set; }
        public double Length { get; set; }
        public double CenterX { get; set; }
        public double CenterY { get; set; }
        public Mat MedialAxis { get; set; }
        public int Area { get; set; }
    }
}
