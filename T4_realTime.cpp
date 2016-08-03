/*
 * T4Bis.cpp
 *
 *  Created on: 17 de abr. de 2016
 *      Author: Jorge
 */
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>

using namespace std;
using namespace cv;

int calcMinMax(vector<vector<Point2f> > objs, vector<Point2f> core,
		vector<float>* sal) {
	float minX = numeric_limits<int>::max();
	float minY = numeric_limits<int>::max();
	float maxX = -numeric_limits<int>::max();
	float maxY = -numeric_limits<int>::max();
	for (int j = 0; j < objs.size(); ++j) {
		vector<Point2f> obj = objs.at(j);
		for (int i = 0; i < 4; ++i) {
			float objX = obj[i].x;
			float objY = obj[i].y;
			float coreX = core[i].x;
			float coreY = core[i].y;
			(objX < minX) ? (minX = objX) : (minX);
			(objX > maxX) ? (maxX = objX) : (maxX);
			(objY < minY) ? (minY = objY) : (minY);
			(objY > maxY) ? (maxY = objY) : (maxY);
			(coreX < minX) ? (minX = coreX) : (minX);
			(coreX > maxX) ? (maxX = coreX) : (maxX);
			(coreY < minY) ? (minY = coreY) : (minY);
			(coreY > maxY) ? (maxY = coreY) : (maxY);
		}
	}
	sal->at(0) = minX;
	sal->at(1) = maxX;
	sal->at(2) = minY;
	sal->at(3) = maxY;
}

Mat compose_core_add(Mat coreCL, Mat addCL, int detectorO, int matcherO) {
	Mat result;
	struct timeval time_init, time_end;
	gettimeofday(&time_init, NULL);

	Mat add, core; // images in Black and white
	cvtColor(addCL, add, CV_RGB2GRAY);
	cvtColor(coreCL, core, CV_RGB2GRAY);

	vector<KeyPoint> keypointsCore, keypointsAdd;

	/* Detects the keypoints with a feature detector */
	if (detectorO == 2) {
		OrbFeatureDetector detector(600);	// Usar NORM_HAMMING en Matcher
		detector.detect(core, keypointsCore);
		detector.detect(add, keypointsAdd);
	} else if (detectorO == 3) {
		SiftFeatureDetector detector(100);	// Usar NORM_L2 en Matcher
		detector.detect(core, keypointsCore);
		detector.detect(add, keypointsAdd);
	} else {
		SurfFeatureDetector detector(300);	// Usar NORM_L2 en Matcher
		detector.detect(core, keypointsCore);
		detector.detect(add, keypointsAdd);
	}

	/* Extracts the descriptors from the features */
	SurfDescriptorExtractor extractor;
	//OrbDescriptorExtractor extractor;
	//SiftDescriptorExtractor extractor;

	Mat descriptorsCore, descriptorsAdd;
	extractor.compute(core, keypointsCore, descriptorsCore);
	extractor.compute(add, keypointsAdd, descriptorsAdd);
	vector<vector<DMatch> > matches;
	/* Finds matches between both images */
	if (matcherO == 1) {
		if (detectorO == 2) {
			BFMatcher matcher(NORM_HAMMING);
			matcher.knnMatch(descriptorsAdd, descriptorsCore, matches, 2);

		} else {
			BFMatcher matcher(NORM_L2);
			matcher.knnMatch(descriptorsAdd, descriptorsCore, matches, 2);
		}
	} else {
		FlannBasedMatcher matcher;
		matcher.knnMatch(descriptorsAdd, descriptorsCore, matches, 2);
	}

	//FlannBasedMatcher matcher;

	/* Eliminate bad matches */
	vector<DMatch> good_matches;
	for (int i = 0; i < (int) matches.size(); i++) {
		float d1 = matches.at(i).at(0).distance;
		float d2 = matches.at(i).at(1).distance;
		if (d1 < 0.45 && d1 / d2 < 0.5) {	//  2nd vecino
			good_matches.push_back(matches.at(i).at(0));
		}
	}

	Mat img_matches;
	drawMatches(add, keypointsAdd, core, keypointsCore, good_matches,
			img_matches);

	/* Localize the object */
	vector<Point2f> picCoreAdd;
	vector<Point2f> picAddCore;
	for (int i = 0; i < (int) good_matches.size(); i++) {
		/* Get the keypoints from the good matches */
		picCoreAdd.push_back(keypointsAdd[good_matches[i].queryIdx].pt);
		picAddCore.push_back(keypointsCore[good_matches[i].trainIdx].pt);
	}

	//Si se pueden considerar imágenes parecidas
	if (good_matches.size() >= 10) {
		Mat H = findHomography(picCoreAdd, picAddCore, CV_RANSAC);

		/* Get the corners from the image_1 ( the object to be "detected" ) */
		vector<Point2f> obj_corners12(4);
		vector<Point2f> core_corners(4);
		obj_corners12[0] = cvPoint(0, 0);
		obj_corners12[1] = cvPoint(add.cols, 0);
		obj_corners12[2] = cvPoint(add.cols, add.rows);
		obj_corners12[3] = cvPoint(0, add.rows);

		core_corners[0] = cvPoint(0, 0);
		core_corners[1] = cvPoint(core.cols, 0);
		core_corners[2] = cvPoint(core.cols, core.rows);
		core_corners[3] = cvPoint(0, core.rows);

		vector<Point2f> scene_corners12(4);
		perspectiveTransform(obj_corners12, scene_corners12, H);

		/* Draw lines between the corners (the mapped object in the scene - image_2 ) */
		line(img_matches, scene_corners12[0] + Point2f(add.cols, 0),
				scene_corners12[1] + Point2f(add.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners12[1] + Point2f(add.cols, 0),
				scene_corners12[2] + Point2f(add.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners12[2] + Point2f(add.cols, 0),
				scene_corners12[3] + Point2f(add.cols, 0), Scalar(0, 255, 0),
				4);
		line(img_matches, scene_corners12[3] + Point2f(add.cols, 0),
				scene_corners12[0] + Point2f(add.cols, 0), Scalar(0, 255, 0),
				4);

		vector<vector<Point2f> > scenes;  
		scenes.push_back(scene_corners12);
		vector<float> minMax(4);
		calcMinMax(scenes, core_corners, &minMax);
		int width = minMax[1] - minMax[0];
		int height = minMax[3] - minMax[2];

		Mat T = Mat::eye(3, 3, CV_64FC1);
		if (minMax[0] < 0) {
			T.at<double>(0, 2) = -minMax[0];
		}
		if (minMax[2] < 0) {
			T.at<double>(1, 2) = -minMax[2];
		}

		/* Use the Homography Matrix to warp the images */
		result = Mat(Size(width, height), CV_32F);
		warpPerspective(coreCL, result, T, result.size(), INTER_LINEAR,
				BORDER_TRANSPARENT);
		warpPerspective(addCL, result, T * H, result.size(), INTER_LINEAR,
				BORDER_TRANSPARENT);
		gettimeofday(&time_end, NULL);
		float ms = (time_end.tv_usec - time_init.tv_usec) / 1000;
		if (ms < 0)
			ms = 1000 + ms;
		cout << "Ha transcurrido " << ms
				<< " milisegundos al juntar una imagen con otra " << endl;

		//imshow("Core", coreCL);
		//imshow("Add", addCL);

		Mat show = result;
		//imshow("Result", show);
		coreCL = result;

		/* Show detected matches */
		Mat matchesShow;
		resize(img_matches, matchesShow,
				Size(img_matches.cols, img_matches.rows));
		//imshow("Good Matches & Object detection 1->2", matchesShow);

	} else {
		result = coreCL;
	}
	return result;

}
int main(int argc, char **argv) {

	//Detector 1 -> surf
	//Detector 2 -> orb
	//Detector 3 -> sift

	//Matcher 1 -> BF
	//Matcher 2 -> Flan

	VideoCapture cap(0);
	Mat core, img;

	int i = 0;
	if (cap.isOpened()) {
		cap >> core;
		int columsSize = 900.0 / (core.cols * 1.0 + core.rows) * core.cols;
		int rowsSize = 900.0 / (core.cols * 1.0 + core.rows) * core.rows;
		resize(core, core, Size(columsSize, rowsSize));
		while (true) {
			i++;
			if (i % 100 == 0) {
				i = 0;
				//coger nueva foto
				cap >> img;
				resize(img, img, Size(columsSize, rowsSize));
				core = compose_core_add(core, img, 1, 1);

			}

			imshow("image", core);

			if (waitKey(1) == 27)
				break; // stop capturing by pressing ESC

		}

	}

}

