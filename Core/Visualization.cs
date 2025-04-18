using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace HoloCrack
{
    public static class Visualization
    {
        /// <summary>
        /// Visualize a segmentation mask by assigning colors to class IDs, resizing,
        /// then alpha-blending onto the original image.
        /// </summary>
        /// <param name="segMask">1D float array (class IDs). Usually [height*width] or [1*height*width].</param>
        /// <param name="maskHeight">The segmentation mask's height (number of rows).</param>
        /// <param name="maskWidth">The segmentation mask's width (number of columns).</param>
        /// <param name="originalImagePath">Path to the original image file.</param>
        /// <param name="resultPath">Where to save the final blended image.</param>
        /// <param name="alpha">Blend factor, e.g. 0.5 for 50% overlay.</param>
        public static void ColorizeAndBlendMask(
            float[] segMask, int maskHeight, int maskWidth,
            string originalImagePath, string resultPath,
            float alpha = 0.5f)
        {
            // 1. Load the original image (BGR order in OpenCvSharp)
            Mat original = Cv2.ImRead(originalImagePath);
            if (original.Empty())
            {
                Console.WriteLine($"Could not load original image: {originalImagePath}");
                return;
            }
            
            // 2. Create a color palette (BGR in OpenCv).
            //    E.g.: index 0=black, 1=red, 2=green, 3=blue...
            var colors = new List<Vec3b>
            {
                new Vec3b(0, 0, 0),   // class 0 = black
                new Vec3b(0, 0, 255), // class 1 = red
                new Vec3b(0, 255, 0), // class 2 = green
                new Vec3b(255, 0, 0), // class 3 = blue
                // Add more colors if you have more classes
            };

            // 3. Build a Mat to hold the colored mask in BGR format
            Mat coloredMask = new Mat(maskHeight, maskWidth, MatType.CV_8UC3, new Scalar(0, 0, 0));

            // segMask is a 1D float array. We'll convert each pixel to an integer classId.
            // Then assign a color from our palette.
            // (If your model is single-class crack vs. background, you might see 0/1 only)
            for (int i = 0; i < segMask.Length; i++)
            {
                int row = i / maskWidth;
                int col = i % maskWidth;

                int classId = (int)segMask[i];
                if (classId >= 0 && classId < colors.Count)
                {
                    coloredMask.Set(row, col, colors[classId]);
                }
                else
                {
                    // Fallback color if classId is out of range
                    coloredMask.Set(row, col, new Vec3b(128, 128, 128));
                }
            }

            // 4. Resize the colored mask to match the original image's size
            Cv2.Resize(
                src: coloredMask,
                dst: coloredMask,
                dsize: new Size(original.Cols, original.Rows),
                fx: 0, fy: 0,
                interpolation: InterpolationFlags.Nearest
            );

            // 5. Blend the colored mask with the original image
            //    alpha = how much of the mask to show
            //    (1 - alpha) = how much of the original image
            Mat overlay = original.Clone();
            Cv2.AddWeighted(coloredMask, alpha, overlay, 1 - alpha, 0, overlay);

            // 6. Save the result
            Cv2.ImWrite(resultPath, overlay);
            Console.WriteLine($"Saved segmentation overlay to {resultPath}");
        }
    }
}
