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
using namespace cv;
using namespace std;

int main(int argc, char **argv) {
	Mat img2 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img1 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	if (img1.empty() || img2.empty()) {
		printf("Can't read one of the images\n");
		return -1;
	}
	resize(img1, img1, Size(), 0.1, 0.1, INTER_CUBIC);
	resize(img2, img2, Size(), 0.1, 0.1, INTER_CUBIC);
	// detecting keypoints
	//between sift surf and brief, we decided to choose surf because we don't need to
	//have invariance in the scale, and surf works better than brief (it's slower though),
	//in orientation changes

	//-- Step 1: Detect the keypoints and generate their descriptors using SURF

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
	vector<vector<DMatch> > matches2;
	vector<DMatch> matchesGood;
	matcher.knnMatch(descriptors1, descriptors2, matches, 2);
	matcher.knnMatch(descriptors2, descriptors1, matches2, 2);

	//Delete bad matches from image1 to image2
	for (int i = 0; i < matches.size(); i++) {
		cout << matches[i][0].distance << endl;
		cout << matches[i][1].distance << endl;
		cout << endl;
		if (matches[i][1].distance < 0.4
				&& matches[i][1].distance - matches[i][0].distance > 0.085) {
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

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(img1.cols, 0);
	obj_corners[2] = cvPoint(img1.cols, img1.rows);
	obj_corners[3] = cvPoint(0, img1.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);
	// drawing the results
	namedWindow("matches", 1);
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, matchesGood, img_matches);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
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

	imshow("matches", img_matches);

	waitKey(0);
}
