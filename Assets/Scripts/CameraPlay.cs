using UnityEngine;
using UnityEngine.UI;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.Collections;
using Emgu.CV.Util;
using System.Text.RegularExpressions;
using static TMPro.SpriteAssetUtilities.TexturePacker_JsonArray;
using Emgu.CV.Features2D;
using Unity.VisualScripting;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Emgu.CV.Flann;

public class CameraPlay : MonoBehaviour
{
    private VideoCapture webcam;
    private Mat frame;
    private bool greenDetected = false;
    public RawImage rawImage;
    public GameObject[] ingredients;
    private Texture2D tex;
    public GameObject cursorObject; 
    private int best=0;
    private GameObject prefab;
    private static List<string> imagePaths = new List<string>(); 
    private bool selected=false;
   List<GameObject> cloned = new List<GameObject>();
    //gerer la cemera
    private void HandleWebcamQueryFrame(object sender, System.EventArgs e)
    {
        if (webcam.IsOpened)
        {
            webcam.Retrieve(frame);
        }
        lock (frame)
        {
            Mat matGrayscale = new Mat(frame.Width, frame.Height, DepthType.Cv8U, 1);
            CvInvoke.CvtColor(frame, matGrayscale, ColorConversion.Bgr2Gray);
            Point targetCenter = DetectColors(frame);
            if(targetCenter != null && targetCenter!=Point.Empty) 
            { 
                CvInvoke.Circle(frame, targetCenter, 30, new MCvScalar(0, 255, 0), -1);
            }
            MoveCursor(targetCenter);
            int image = 1;
            best = 0;
            int c = 0;
            //---------------------------------------------------------------
            //Appeler canny pour trouver le rectangle
            bool FindRect = CannyRectangleDetection(matGrayscale);
            if (FindRect)
            { 
                instantiate();
            }
        }
        System.Threading.Thread.Sleep(200);
    }
    public void instantiate()
    {
        prefab = ingredients[UnityEngine.Random.Range(0, ingredients.Length)];
        Vector2 position=cursorObject.transform.position;
        var obj=Instantiate(prefab, position,Quaternion.identity);
        cloned.Add(obj);
        StartCoroutine(wait());
    }
    IEnumerator wait()
    {
        yield return new WaitForSeconds(2);
    }
    void Start()
    {
        Debug.Log("starting webcam");
        webcam = new Emgu.CV.VideoCapture(0, VideoCapture.API.DShow);
        frame = new Mat();
        webcam.ImageGrabbed += new System.EventHandler(HandleWebcamQueryFrame);
        webcam.Start();
    }

    // Update is called once per frame
    void Update()
    {
        if (webcam.IsOpened)
        {
            // send event that an image has been acquired
            bool grabbed = webcam.Grab();
        }

        DisplayFrameOnPlane();//manque dans le sujet
        
    }
    void OnDestroy()
    {
        Debug.Log("entering destroy");

        if (webcam != null)
        {

            Debug.Log("sleeping");
            System.Threading.Thread.Sleep(60);
            // close camera
            webcam.Stop();
            webcam.Dispose();
        }

        Debug.Log("Destroying webcam");
    }
    //canny rectangle detection
    public bool CannyRectangleDetection(Mat grayMat)
    {
        Mat edges = new Mat();
        bool FindRect = false;
        CvInvoke.Canny(grayMat, edges, 170, 200);

        // Find contours in the edges
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        CvInvoke.FindContours(edges, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
        for (int i = 0; i < contours.Size; i++)
        {
            using (VectorOfPoint contour = contours[i])
            {
                VectorOfPoint approx = new VectorOfPoint();
                CvInvoke.ApproxPolyDP(contour, approx, CvInvoke.ArcLength(contour, true) * 0.04, true);

                // Check if rectangle
                if (CvInvoke.ContourArea(approx) > 200 && CvInvoke.IsContourConvex(approx) && approx.Size == 4)
                {
                    RotatedRect rect = CvInvoke.MinAreaRect(approx);
                    if (rect.Size.Width * rect.Size.Height > 1000)
                    {
                        // Draw the rectangle 
                        CvInvoke.Polylines(frame, new VectorOfVectorOfPoint(approx), true, new MCvScalar(0,0 ,255 ), 2);
                        FindRect = true;
                    }
                }
            }
        }
        return FindRect;
    }
    private void DisplayFrameOnPlane()
    {
        if (frame.IsEmpty) return;

        int width = (int)rawImage.rectTransform.rect.width;
        int height = (int)rawImage.rectTransform.rect.height;
        if (tex != null)
        {
            Destroy(tex);
            tex = null;
        }

        // creating new texture to hold our frame
        tex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        CvInvoke.Resize(frame, frame, new System.Drawing.Size(width, height));
        CvInvoke.CvtColor(frame, frame, ColorConversion.Bgr2Rgba);
        CvInvoke.Flip(frame, frame, FlipType.Vertical);
        tex.LoadRawTextureData(frame.ToImage<Rgba, byte>().Bytes);
        tex.Apply();
        rawImage.texture = tex;
        

    }
    Point DetectColors(Mat frame)
    {
        Mat hsvFrame = new Mat();
        CvInvoke.CvtColor(frame, hsvFrame, ColorConversion.Bgr2Hsv);
        ScalarArray min = new ScalarArray(new MCvScalar(30, 100, 40));
        ScalarArray max = new ScalarArray(new MCvScalar(90, 200, 255));

        // Threshold the frame 
        Mat greenMask = new Mat();
        CvInvoke.InRange(hsvFrame, min, max,greenMask);

        // Find contours 
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        CvInvoke.FindContours(greenMask, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

        // Find the contour with the largest area 
        double maxArea = 0;
        int maxAreaIndex = -1;

        for (int i = 0; i < contours.Size; i++)
        {
            double area = CvInvoke.ContourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                maxAreaIndex = i;
            }
        }
        // Get the center
        Point ObjectCenter = new Point(0, 0);
        if (maxAreaIndex != -1)
        {
            Moments moments = CvInvoke.Moments(contours[maxAreaIndex]);
            int cX = (int)(moments.M10 / moments.M00);
            int cY = (int)(moments.M01 / moments.M00);
            ObjectCenter = new Point(cX, cY);
            CvInvoke.Circle(frame, ObjectCenter, 5, new MCvScalar(0, 255, 0), -1);
            //CvInvoke.Circle(frame, redObjectCenter, 10, new MCvScalar(255, 255, 0));
            greenDetected = true;
            Debug.Log("detected");
        }
        else
        {
            greenDetected = false;
        }
        hsvFrame.Dispose();
        greenMask.Dispose();
        contours.Dispose();

        return ObjectCenter;
    }
    //fonction Qui sert à deplacer le curseur
    void MoveCursor(Point greenObjectCenter)
    {
        if (greenDetected)
        {
            // get coordinates
            double cursorX = -(double)((greenObjectCenter.X / webcam.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth))-0.5);
            double cursorY = -(double)((greenObjectCenter.Y/ webcam.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight))-0.5);
            //Si on veut se positionner sur tout le screen
            //cursorX *= (Screen.width / rawImage.rectTransform.rect.width*2);
            //cursorY *= Screen.height / rawImage.rectTransform.rect.height;
            cursorY += 0.3;
            // faire bouger le curseur
            cursorObject.transform.position = Vector3.Lerp(cursorObject.transform.position, new Vector3((float)cursorX, (float)cursorY, 0), Time.deltaTime * 8);
        }
    }
    //detruire les objets 
    public void DestroyAllObjects()
    {
        foreach (GameObject obj in cloned)
        {
            Destroy(obj);
        }
    }
}


