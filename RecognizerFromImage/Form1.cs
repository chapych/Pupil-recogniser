using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace RecognizerFromImage
{
    public class Matrix
    {
        int[,] data;

        public Matrix(Tuple<int, int>[] input)
        {
            int height = input.GetLength(0);
            data = new int[5, 6];
            for (int i = 0; i < 5; i++)
            {
                int x = input[i].Item1;
                int y = input[i].Item2;
                data[i, 0] = x * x;
                data[i, 1] = x * y;
                data[i, 2] = y * y;
                data[i, 3] = x;
                data[i, 4] = y;
                data[i, 5] = 1;
            }
        }

        public Matrix(Matrix matrix, int column, int raw)
        {
            int[,] input = matrix.data;
            int height = input.GetLength(0);
            int width = input.GetLength(1);
            int shiftI = 0;
            int shiftJ = 0;
            data = new int[--height, --width];
            for (int i = 0; i < width; i++)
            {
                if (i == column) shiftI++;
                shiftJ = 0;
                for (int j = 0; j < height; j++)
                {
                    if (j == raw) shiftJ++;
                    data[j, i] = input[j + shiftJ, i + shiftI];
                }
            }
        }

        public Matrix(Matrix matrix, int column)
        {
            int[,] input = matrix.data;
            int height = input.GetLength(0);
            int width = input.GetLength(1);
            data = new int[height, --width];
            int shift = 0;
            for (int i = 0; i < width; i++)
            {

                if (i == column) shift++;
                for (int j = 0; j < height; j++)
                {
                    data[j, i] = input[j, i + shift];
                }
            }
        }

        float ThreeDimDeterminant(Matrix matrix)
        {
            int[,] data = matrix.data;
            if (matrix.data.GetLength(0) != 3) throw new ArgumentException();
            float determinant = 0.0f;
            int raw = 0;
            for (int i = 0; i < 3; i++)
            {
                int frstEl = data[(raw + 1) % 3, (i + 1) % 3];
                int scndEl = data[(raw + 2) % 3, (i + 2) % 3];
                int negFrstEl = data[(raw + 1) % 3, (i + 2) % 3];
                int negScndEl = data[(raw + 2) % 3, (i + 1) % 3];
                int coeff = frstEl * scndEl - negFrstEl * negScndEl;
                determinant += data[raw, i] * coeff;
            }
            return determinant;
        }

        public float Determinant()
        {
            float result = 0.0f;
            if (this.data.GetLength(0) != 3)
            {
                for (int i = 0; i < this.data.GetLength(0); i++)
                {
                    Matrix subMatrix = new Matrix(this, i, 0);
                    result += (float)Math.Pow(-1, i) * this.data[0, i] * subMatrix.Determinant();

                }

            }
            if (data.GetLength(0) == 3) result = ThreeDimDeterminant(this);
            return result;
        }

    }

    public partial class Form1 : Form
    {
        private static CascadeClassifier classifier = new CascadeClassifier("haarcascade_eye.xml");
        public static Image<Gray, byte> pupil;

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            DialogResult res = openFileDialog1.ShowDialog();
            if (res == DialogResult.OK)
            {
                string path = openFileDialog1.FileName;

                pictureBox1.Image = Image.FromFile(path);

                Bitmap bitmap = new Bitmap(pictureBox1.Image);

                Image<Bgr, byte> grayImage = new Image<Bgr, byte>(path);

                Rectangle[] eyes = classifier.DetectMultiScale(grayImage, 1.1, 4, Size.Empty, Size.Empty);

                for (int i = 0; i < eyes.Length; i++)
                {
                    eyes[i].Inflate(new Size(0, -eyes[i].Height / 4));
                }

                foreach (Rectangle eye in eyes)
                {
                    using (Graphics graphics = Graphics.FromImage(bitmap))
                    {
                        using (Pen pen = new Pen(Color.BlueViolet, 4))
                        {
                            graphics.DrawRectangle(pen, eye);
                        }
                    }

                }

                Rectangle cutContour = FindMaxRectangle(eyes);

                Mat cutImage = new Mat(grayImage.Mat, cutContour);
                pupil = cutImage.ToImage<Gray, Byte>();
                pictureBox1.Image = bitmap;
            }
            else MessageBox.Show("No image", "error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        private async void button2_Click(object sender, EventArgs e)
        {
            DialogResult res = DialogResult.OK;

            if (res == DialogResult.OK)
            {

                Mat mat = ConvertToBlackWhiteMatrix(pupil);

                int rows = mat.Rows;
                int cols = mat.Cols;

                mat = KMean(mat, rows);

                Image<Gray, byte> image = Normalization(mat, rows, cols);

                image = Morph(image, MorphOp.Open);

                image = Morph(image, MorphOp.Close);

                CvInvoke.Threshold(image, image, 200, 255, ThresholdType.Binary);

                mat.ConvertTo(mat, Emgu.CV.CvEnum.DepthType.Cv8U);

                Tuple<int, int>[] array = FindRandomPoints(image, rows, cols);

                PointF[] ellipseData = array.Select(x => new PointF(x.Item2, x.Item1))
                                            .ToArray();

                Ellipse ellipse = Emgu.CV.PointCollection.EllipseLeastSquareFitting(ellipseData);

                Image<Rgb, byte> iimage = image.Convert<Rgb, Byte>();

                foreach (Tuple<int, int> el in array)
                {
                    if (el == null) break;
                    CvInvoke.Circle(iimage, new Point(el.Item1, el.Item2), 0,
                                    new Bgr(Color.Red).MCvScalar, 1);
                }

                iimage.Draw(ellipse, new Rgb(Color.BlueViolet), 1, LineType.EightConnected, 0);

                ImageViewer viewer = new ImageViewer();
                viewer.Image = iimage;
                viewer.Show();
            }
            else MessageBox.Show("No image", "error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        public Rectangle FindMaxRectangle(Rectangle[] rectsngles)
        {
            int max = Int32.MinValue;
            Rectangle result = new Rectangle(1, 1, 1, 1);
            foreach (Rectangle rectangle in rectsngles)
            {
                if (rectangle.Height > max)
                {
                    result = rectangle;
                    max = rectangle.Height;
                }
            }

            return result;
        }

        public Mat ConvertToBlackWhiteMatrix(string path)
        {
            Image<Gray, byte> grayImage = new Image<Gray, byte>(path);

            Mat mat = grayImage.Mat;

            //var gaussImage = new Mat(rows, col, Emgu.CV.CvEnum.DepthType.Cv32F, 1); // in case resolution is low
            //CvInvoke.GaussianBlur(tmp, gaussImage, new Size(0, 0), 500);
            //CvInvoke.AddWeighted(tmp, 2.8, gaussImage, -2.3, 0, tmp);

            return mat;
        }

        public Mat ConvertToBlackWhiteMatrix(Image<Gray, byte> grayImage)
        {
            Mat mat = grayImage.Mat;

            //var gaussImage = new Mat(rows, col, Emgu.CV.CvEnum.DepthType.Cv32F, 1); // in case resolution is low
            //CvInvoke.GaussianBlur(tmp, gaussImage, new Size(0, 0), 500);
            //CvInvoke.AddWeighted(tmp, 2.8, gaussImage, -2.3, 0, tmp);

            return mat;
        }

        public Mat KMean(Mat mat, int rows)
        {
            mat = mat.Clone();

            mat = mat.Reshape(0, 1);

            Mat centers = new Mat();

            mat.ConvertTo(mat, Emgu.CV.CvEnum.DepthType.Cv32F);

            MCvTermCriteria criteria = new MCvTermCriteria(10, 0.001);

            double a = CvInvoke.Kmeans(mat, 2, mat, criteria, 2, KMeansInitType.PPCenters, centers);

            mat = mat.Reshape(0, rows);

            return mat;
        }

        public Image<Gray, Byte> Morph(Image<Gray, Byte> image, MorphOp smth)
        {
            Mat mat = image.Mat;

            Mat kernel = CvInvoke.GetStructuringElement(Emgu.CV.CvEnum.ElementShape.Rectangle, new Size(10, 10), new Point(-1, -1));

            CvInvoke.MorphologyEx(mat, mat, smth, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(1.0));

            return image;
        }

        public Image<Gray, Byte> Normalization(Mat mat, int rows, int cols)
        {
            Image<Gray, byte> image = mat.ToImage<Gray, Byte>();
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {

                    if (image[i, j].Equals(new Gray(0))) image[i, j] = new Gray(0);
                    if (image[i, j].Equals(new Gray(1))) image[i, j] = new Gray(255);
                }
            return image;
        }

        public Image<Gray, Byte> Average(Image<Gray, Byte> image, int rows, int cols)
        {
            Image<Gray, byte> newImage = new Image<Gray, Byte>(cols, rows);
            int start = 1;

            for (int i = start; i < rows - start; i++)
                for (int j = start; j < cols - start; j++)
                {
                    double intens = (image[i - start, j].Intensity + image[i + start, j].Intensity + image[i, j - start].Intensity + image[i, j + start].Intensity) / 4;
                    if (intens > 122) intens = 255;
                    else intens = 0;
                    newImage[i, j] = new Gray(intens);
                }

            return newImage;
        }

        public Tuple<int, int>[] FindRandomPoints(Image<Gray, Byte> image, int cols, int rows)
        {
            int occupied = 0;
            int possibility = 5;
            int shift = 10;
            Tuple<int, int>[] array = new Tuple<int, int>[100];
            while (occupied < 100)
            {
                for (int i = 1; i < cols - 1; i++)
                    for (int j = 1; j < rows - 1; j++)
                    {
                        if (i >= cols - 1 || j >= rows - 1)
                        {
                            i = 1;
                            j = 1;
                            continue;
                        }
                        bool flag1 = !image[i, j].Equals(image[i, j - 1]) || !image[i, j].Equals(image[i - 1, j]);
                        bool flag2 = !image[i, j].Equals(image[i, j + 1]) || !image[i, j].Equals(image[i + 1, j]);
                        if (flag1 || flag2)
                        {
                            Random r = new Random();
                            if (r.Next(11) > possibility)
                            {
                                if (occupied < 100)
                                {
                                    array[occupied] = Tuple.Create(j, i);
                                    i = i + shift;
                                    j = j + shift;
                                    occupied++;
                                    //possibility++;
                                }
                            }

                        }
                    }
            }
            foreach (Tuple<int, int> el in array)
            {
                if (el == null) break;
                Console.WriteLine(el.Item1 + " " + el.Item2);
            }

            return array;
        }
        //public void FindCircles()
        //{
        //    //image = output.Reshape(0, rows).ToImage<Gray, Byte>(); //обнаружение кругов

        //    //var circles = CvInvoke.HoughCircles(image, HoughModes.Gradient, 1, 50, 100, 10, 10, 100000);
        //    //foreach (CircleF circle in circles)
        //    //    CvInvoke.Circle(image, Point.Round(circle.Center), (int)circle.Radius,
        //    //        new Bgr(Color.Brown).MCvScalar, 2);
        //}

        public float[] FindEllipseProp(Tuple<int, int>[] input)
        {
            //input = new Tuple<int, int>[] { Tuple.Create(1, 2), Tuple.Create(2, 2), Tuple.Create(3, 1), Tuple.Create(4, 3), Tuple.Create(5, 1) };
            float[] result = new float[6];
            Matrix data = new Matrix(input);
            for (int i = 0; i < 6; i++)
            {
                Matrix subMatrix = new Matrix(data, i);
                result[i] = (float)Math.Pow(-1, i) * subMatrix.Determinant();
            }

            return result;
        }

        public Ellipse FindEllipse(float[] points) //todo later
        {
            float a = points[0];
            float b = points[1];
            float c = points[2];
            float d = points[3];
            float e = points[4];
            float f = points[5];
            PointF center = new PointF((2 * c * d - b * e) / (b * b - 4 * a * c), (2 * a * e - b * d) / (b * b - 4 * a * c));
            float element = -1 * (float)1 / (b * b - 4 * a * c);
            float el1 = element * (float)Math.Sqrt(2 * (a * e * e + c * d * d - b * d * e + b * b * f - 4 * a * c * f) * (a + c + Math.Sqrt(a * a + c * c - 2 * a * c + b * b)));
            float el2 = element * (float)Math.Sqrt(2 * (a * e * e + c * d * d - b * d * e + b * b * f - 4 * a * c * f) * (a + c - Math.Sqrt(a * a + (c * c - 2 * a * c + b * b))));
            SizeF size = new SizeF(el1, el2);
            Ellipse ellipse = new Ellipse(center, size, (float)Math.Atan((1 / b) * (c - a - Math.Sqrt(a * a + c * c - 2 * a * c + b * b))));
            return ellipse;
        }

    }
}
