#include <iostream>
 #include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{

    VideoCapture cap(0); //capture the video from webcam

    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }
    // Create control panel window

    int iLowH = 0;
    int iHighH = 29;
    int iLowS = 61;
    int iHighS = 199;
    int iLowV = 88;
    int iHighV = 255;
    //Create trackbars in "Results" window
    namedWindow("Results", CV_WINDOW_AUTOSIZE); //create a window called "Control"
    createTrackbar("LowH", "Results", &iLowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Results", &iHighH, 179);
    createTrackbar("LowS", "Results", &iLowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Results", &iHighS, 255);
    createTrackbar("LowV", "Results", &iLowV, 255);//Value (0 - 255)
    createTrackbar("HighV", "Results", &iHighV, 255);

    int iMinBlobSize = 1;
    createTrackbar("MinBlobSize*1000", "Results", &iMinBlobSize, 100);



    while (true)
    {
        /// Read a new frame from video
        ///

        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal);
        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        /// Conversion from BGR to HSV color space

        Mat imgHSV;
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        Mat imgThresholded;
         inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
         imshow("Thresholded Image", imgThresholded); //show the thresholded image

         /// median blurr

         medianBlur(imgThresholded, imgThresholded, 3);
        imshow("median Blur Image", imgThresholded); //show the thresholded image

         /// morphological opening (removes small objects from the foreground)

         erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
         dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

         /// morphological closing (removes small holes from the foreground)

         dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
         erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
         imshow("corrected Image", imgThresholded); //show the thresholded image
         // 1. Get the blobs (i.e. their contour)

         //contour.size();
         vector<vector<Point>> contours;
         vector<Vec4i> hierarchy;
         findContours( imgThresholded,
                       contours,
                       hierarchy,
                       CV_RETR_TREE,
                       CV_CHAIN_APPROX_SIMPLE,
                       Point(0, 0) );
         int ncont;
         ncont = contours.size();
         //Rect fingerRect[ncont];
         RotatedRect fingerRectRotated[ncont], smallRect[ncont];

        for( int i = 0; i < ncont; i++ )
         {

        fingerRectRotated[i] = minAreaRect(contours[i]);

         Point2f vertice[4], center1;
         Point2f vertices[4], center2;
         float angle1;

         double width1, height1;

         angle1 =  fingerRectRotated[i].angle;
         center1 = fingerRectRotated[i].center;

         width1 = fingerRectRotated[i].size.width*0.9;
         height1 = fingerRectRotated[i].size.height*0.9;

          fingerRectRotated[i].points(vertice);

          if (fingerRectRotated[i].size.width > fingerRectRotated[i].size.height)
          {
         center2.x =  (int)round(center1.x + width1 *1/3* cos(angle1 * CV_PI / 180.0));
         center2.y =  (int)round(center1.y + width1*1/3* sin(angle1 * CV_PI / 180.0));


         smallRect[i] = RotatedRect(center2, Size2f(1.5*height1,height1), angle1);
          }
          else
          {
              angle1 = 180 - fingerRectRotated[i].angle;
              center2.y =  (int)round(center1.y + height1 *1/3* cos(angle1 * CV_PI / 180.0));
              center2.x =  (int)round(center1.x + height1 *1/3* sin(angle1 * CV_PI / 180.0));


              smallRect[i] = RotatedRect(center2, Size2f(width1,1.3*width1), 180-angle1);
          }

         smallRect[i].points(vertices);

             for( int j = 0; j < 4; j++ )
             {
             line(imgOriginal, vertices[j], vertices[(j+1)%4], Scalar(0,255,0),1,8);
 //            circle(imgOriginal, vertices[0], 5, Scalar(255,0,0),1, 8);
 //            circle(imgOriginal, vertices[1], 5, Scalar(0,255,0),1, 8);
 //            circle(imgOriginal, vertices[2], 5, Scalar(0,0,255),1, 8);
 //            circle(imgOriginal, vertices[3], 5, Scalar(255,255,255),1, 8);
             circle(imgOriginal, center2, 2, Scalar(255,0,0),2,8);

            }
         }

         // 2. Get the moments
         vector<Moments> mu(contours.size() );
         for( int i = 0; i < contours.size(); i++ )
         {
             mu[i] = moments( contours[i], false );
         }

        //  3. Get the mass centers (centroids)
         //
         vector<Point2f> mc( contours.size() );
         for( int i = 0; i < contours.size(); i++ )
         {
             mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
         }

        // 4. Draw blobs
         //
         RNG rng(12345);
         Mat blobImg = Mat::zeros( imgThresholded.size(), CV_8UC3 );
         for( int i = 0; i< contours.size(); i++ )
         {
             Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
             drawContours( blobImg, contours, i, color, 2, 8, hierarchy, 0, Point() );
           //  circle( blobImg, mc[i], 4, color, -1, 8, 0 );
         }
       //  imshow( "blobImg", blobImg );

        // 5. Blob filtering
         //
         vector<vector<Point>> contours_filt;
         vector<Vec4i> hierarchy_filt;
         vector<Moments> mu_filt;
         vector<Point2f> mc_filt;

       // Filter blobs w.r.t. to size
         int minBlobArea = iMinBlobSize*1000; // Minimum Pixels for valid blob
         for( int i = 0; i< contours.size(); i++ )
         {
             if (mu[i].m00 > minBlobArea)
             {
                 contours_filt.push_back(contours[i]);
                 hierarchy_filt.push_back(hierarchy[i]);
                 mu_filt.push_back(mu[i]);
                 mc_filt.push_back(mc[i]);
             }
         }

        // 6. Draw filtered blobs
         //
         Mat blobImg_filt = Mat::zeros( imgThresholded.size(), CV_8UC3 );;
         for( int i = 0; i< contours_filt.size(); i++ )
         {
             Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
             drawContours( blobImg_filt, contours_filt, i, color, 2, 8, hierarchy_filt, 0, Point() );
           //  circle( blobImg_filt, mc_filt[i], 4, color, -1, 8, 0 );
         //   drawContours( imgOriginal, contours_filt, i, color, 2, 8, hierarchy_filt, 0, Point() );
             circle( imgOriginal, mc_filt[i], 4, color, -1, 8, 0 );
         }
        //imshow( "blobImg_filt", blobImg_filt );
         imshow("Results", imgOriginal);

         if (waitKey(30) == 27) //wait for 'esc' key
         {
             cout << "esc key is pressed by user" << endl;
             break;
         }
     }
     return 0;
 }
