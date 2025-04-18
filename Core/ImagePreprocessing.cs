using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace HoloCrack.Core
{
    public static class ImagePreprocessing
    {
        private static readonly float[] mean = { 72.39239876f, 82.90891754f, 73.15835921f };
        private static readonly float[] std = { 1f, 1f, 1f };

        /// <summary>
        /// Loads an image from <paramref name="imagePath"/>,
        /// resizes it to (inputWidth, inputHeight), converts BGR→RGB,
        /// normalizes by mean/std, and returns a DenseTensor in shape [1, 3, H, W].
        /// </summary>
        public static DenseTensor<float> LoadAndPrepareImage(
            string imagePath,
            int inputWidth = 1024,
            int inputHeight = 1024)
        {
            Mat img = Cv2.ImRead(imagePath, ImreadModes.Color);
            if (img.Empty())
                throw new FileNotFoundException($"Could not load image: {imagePath}");

            Mat resized = new Mat();
            Cv2.Resize(img, resized, new Size(inputWidth, inputHeight));
            Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

            return NormalizeImage(resized);
            
        }

        private static DenseTensor<float> NormalizeImage(Mat image)
        {
            int height = image.Rows;
            int width = image.Cols;
            float[] chwData = new float[3 * height * width];

            int offsetR = 0 * height * width;
            int offsetG = 1 * height * width;
            int offsetB = 2 * height * width;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Vec3b pixel = image.Get<Vec3b>(y, x);

                    float r = (pixel.Item0 - mean[0]) / std[0];
                    float g = (pixel.Item1 - mean[1]) / std[1];
                    float b = (pixel.Item2 - mean[2]) / std[2];

                    int idx = y * width + x;
                    chwData[offsetR + idx] = r;
                    chwData[offsetG + idx] = g;
                    chwData[offsetB + idx] = b;
                }
            }

            var NormalizeImage = new DenseTensor<float>(new[] { 1, 3, height, width });
            chwData.AsSpan().CopyTo(NormalizeImage.Buffer.Span);

            return NormalizeImage;
        }
    }
}
