#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv2/legacy/legacy.hpp>
#include <sys/time.h>
using namespace cv;
using namespace std;

Mat panoram(Mat img1color, Mat img2color) {
    Mat img1, img2;

    cvtColor(img1color, img1, CV_BGR2GRAY);
    cvtColor(img2color, img2, CV_BGR2GRAY);
    // detecting keypoints
    //between sift surf and brief, we decided to choose surf because we don't need to
    //have invariance in the scale, and surf works better than brief (it's slower though),
    //in orientation changes

    //-- Step 1: Detect the keypoints and generate their descriptors using SURF
    struct timeval time_init, time_end;
    gettimeofday(&time_init, NULL);

    SurfFeatureDetector detector(400);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher = BFMatcher(NORM_L2, false);
    // FlannBasedMatcher matcher;

    vector<vector<DMatch> > matches;
    vector<DMatch> matchesGood;
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    //Delete bad matches from image1 to image2
    for (int i = 0; i < matches.size(); i++) {
        if (/*matches[i][1].distance < 0.43
                && */matches[i][1].distance - matches[i][0].distance > 0.08) {
            matchesGood.push_back(matches[i][0]);
        }
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (int i = 0; i < matchesGood.size(); i++) {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints1[matchesGood[i].queryIdx].pt);
        scene.push_back(keypoints2[matchesGood[i].trainIdx].pt);
    }

    Mat homography = findHomography(obj, scene, CV_RANSAC);

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0, 0);
    obj_corners[1] = cvPoint(img1.cols, 0);
    obj_corners[2] = cvPoint(img1.cols, img1.rows);
    obj_corners[3] = cvPoint(0, img1.rows);
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, homography);
    //sacar tamaño de la imagen

    cv::Mat result;

    if (scene_corners[0].x < 0) {
        cout << "normal" << endl;
        cout << obj_corners[0] << endl;
               cout << obj_corners[1] << endl;
               cout << obj_corners[2] << endl;
               cout << obj_corners[3] << endl;
               cout << scene_corners[0] << endl;
               cout << scene_corners[1] << endl;
               cout << scene_corners[2] << endl;
               cout << scene_corners[3] << endl;
        Mat homography = findHomography(scene, obj, CV_RANSAC);
        // drawing the results
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, matchesGood,
                img_matches);
        line(img_matches, scene_corners[0] + Point2f(img1.cols, 0),
                scene_corners[1] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[1] + Point2f(img1.cols, 0),
                scene_corners[2] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[2] + Point2f(img1.cols, 0),
                scene_corners[3] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[3] + Point2f(img1.cols, 0),
                scene_corners[0] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);

        //-- Show detected matches
        imshow("Good Matches & Object detection", img_matches);
        warpPerspective(img2color, result, homography,
                cv::Size(img2color.cols +img1color.cols, img2color.rows+img1color.rows));
        cv::Mat half(result, cv::Rect(0, 0, img1color.cols, img1color.rows));
        img1color.copyTo(half);
    	/* To remove the black portion after stitching, and confine in a rectangular region*/

    		// vector with all non-black point positions
    		std::vector<cv::Point> nonBlackList;
    		nonBlackList.reserve(result.rows*result.cols);

    		// add all non-black points to the vector
    		// there are more efficient ways to iterate through the image
    		for(int j=0; j<result.rows; ++j)
    				for(int i=0; i<result.cols; ++i)
    				{
    						// if not black: add to the list
    						if(result.at<cv::Vec3b>(j,i) != cv::Vec3b(0,0,0))
    						{
    								nonBlackList.push_back(cv::Point(i,j));
    						}
    				}

    		// create bounding rect around those points
    		cv::Rect bb = cv::boundingRect(nonBlackList);

        return result(bb);
    } else {
        cout << "vuelta" << endl;
        cout << obj_corners[0] << endl;
               cout << obj_corners[1] << endl;
               cout << obj_corners[2] << endl;
               cout << obj_corners[3] << endl;
               cout << scene_corners[0] << endl;
               cout << scene_corners[1] << endl;
               cout << scene_corners[2] << endl;
               cout << scene_corners[3] << endl;
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, matchesGood,
                img_matches);
        line(img_matches, scene_corners[0] + Point2f(img1.cols, 0),
                scene_corners[1] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[1] + Point2f(img1.cols, 0),
                scene_corners[2] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[2] + Point2f(img1.cols, 0),
                scene_corners[3] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[3] + Point2f(img1.cols, 0),
                scene_corners[0] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);

        //-- Show detected matches
        imshow("Good Matches & Object detection", img_matches);
        warpPerspective(img1color, result, homography,
                cv::Size(img1color.cols + img2color.cols, img1color.rows+img2color.rows));

        cv::Mat half(result, cv::Rect(0, 0, img2color.cols, img2color.rows));
        img2color.copyTo(half);
    	// vector with all non-black point positions
        		std::vector<cv::Point> nonBlackList;
        		nonBlackList.reserve(result.rows*result.cols);

        		// add all non-black points to the vector
        		// there are more efficient ways to iterate through the image
        		for(int j=0; j<result.rows; ++j)
        				for(int i=0; i<result.cols; ++i)
        				{
        						// if not black: add to the list
        						if(result.at<cv::Vec3b>(j,i) != cv::Vec3b(0,0,0))
        						{
        								nonBlackList.push_back(cv::Point(i,j));
        						}
        				}

        		// create bounding rect around those points
        		cv::Rect bb = cv::boundingRect(nonBlackList);
        return result(bb);
    }
    gettimeofday(&time_end, NULL);
    float ms = (time_end.tv_usec - time_init.tv_usec) / 1000;
    cout << "Ha transcurrido " << ms
            << " milisegundos al juntar una imagen con otra " << endl;

}
int main(int argc, char **argv) {
    Mat img0 = imread(argv[0]);

    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
    Mat img3 = imread(argv[3]);
    Mat img4 = imread(argv[4]);
    Mat img5 = imread(argv[5]);

    if (img1.empty() || img2.empty()) {
        printf("Can't read one of the images\n");
        return -1;
    }

    float factor_size = 800.0 / (img1.cols * 1.0 + img1.rows);
    resize(img1, img1, Size(), factor_size, factor_size, INTER_CUBIC);
    resize(img2, img2, Size(), factor_size, factor_size, INTER_CUBIC);
    resize(img3, img3, Size(), factor_size, factor_size, INTER_CUBIC);
    resize(img4, img4, Size(), factor_size, factor_size, INTER_CUBIC);
    resize(img5, img5, Size(), factor_size, factor_size, INTER_CUBIC);

    Mat result = panoram(img1, img2);

    imshow("1", result);
Mat result2 = panoram(result, img3);
   imshow("2", result2);/*

    Mat result3 = panoram(result2, img4);
    imshow("3", result3);

    Mat result4 = panoram(result3, img5);
    imshow("4", result4);*/

    waitKey(0);
}
