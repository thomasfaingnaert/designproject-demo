#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

void plotHistograms(const std::string& title, const cv::Mat &frame)
{
	cv::Mat frameHsv;
	cv::cvtColor(frame, frameHsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_planes;
	cv::split(frameHsv, hsv_planes);
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	cv::Mat h_hist, s_hist, v_hist;
	cv::calcHist(&hsv_planes[0], 1, 0, cv::Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&hsv_planes[1], 1, 0, cv::Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&hsv_planes[2], 1, 0, cv::Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for H,S,V
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::normalize(h_hist, h_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(s_hist, s_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	cv::normalize(v_hist, v_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(h_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(s_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(v_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}

	cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
	cv::imshow(title, histImage);
}

void detectFaceOpenCVDNN(cv::dnn::Net& net, const cv::Mat &inFrame, cv::Mat &outFrame)
{
	int frameHeight = inFrame.rows;
	int frameWidth = inFrame.cols;

	cv::Mat inputBlob = cv::dnn::blobFromImage(inFrame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);

	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");

	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > 0.7)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

			cv::rectangle(outFrame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);

			/*
			if (0 <= x1 && 0 <= y1 && x2 < frameOpenCVDNN.cols && y2 < frameOpenCVDNN.rows)
			{
				cv::Rect face = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
				plotHistograms("Face histogram", frameOpenCVDNN(face));
			}
			*/

		}
	}
}

const int HAND_PAIRS[20][2] =
{
	{0,1}, {1,2}, {2,3}, {3,4},         // thumb
	{0,5}, {5,6}, {6,7}, {7,8},         // index
	{0,9}, {9,10}, {10,11}, {11,12},    // middle
	{0,13}, {13,14}, {14,15}, {15,16},  // ring
	{0,17}, {17,18}, {18,19}, {19,20}   // small
};

const int POSE_PAIRS[17][2] =
{
	{0,1}, {1,2}, {1,5}, {2,3}, {5,6}, {3,4}, {6,7}
};

void detectHandKeypoints(cv::dnn::Net& net, cv::Mat &frame)
{
	float thresh = 0.01;

	cv::Mat frameCopy = frame.clone();
	int frameWidth = frame.cols;
	int frameHeight = frame.rows;

	int nPoints = 22;

	float aspect_ratio = frameWidth / (float)frameHeight;
	int inHeight = 368;
	int inWidth = (int(aspect_ratio*inHeight) * 8) / 8;
	
	cv::Mat inpBlob = cv::dnn::blobFromImage(frame, 1.0 / 255, cv::Size(inWidth, inHeight), cv::Scalar(0, 0, 0), false, false);

	net.setInput(inpBlob);

	cv::Mat output = net.forward();

	int H = output.size[2];
	int W = output.size[3];

	// find the position of the body parts
	std::vector<cv::Point> points(nPoints);
	for (int n = 0; n < nPoints; n++)
	{
		// Probability map of corresponding body's part.
		cv::Mat probMap(H, W, CV_32F, output.ptr(0, n));
		cv::resize(probMap, probMap, cv::Size(frameWidth, frameHeight));

		cv::Point maxLoc;
		double prob;
		cv::minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
		if (prob > thresh)
		{
			circle(frameCopy, cv::Point((int)maxLoc.x, (int)maxLoc.y), 8, cv::Scalar(0, 255, 255), -1);
			cv::putText(frameCopy, cv::format("%d", n), cv::Point((int)maxLoc.x, (int)maxLoc.y), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

		}
		points[n] = maxLoc;
	}

	int nPairs = sizeof(HAND_PAIRS) / sizeof(HAND_PAIRS[0]);

	for (int n = 0; n < nPairs; n++)
	{
		// lookup 2 connected body/hand parts
		cv::Point2f partA = points[HAND_PAIRS[n][0]];
		cv::Point2f partB = points[HAND_PAIRS[n][1]];

		if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
			continue;

		line(frame, partA, partB, cv::Scalar(0, 255, 255), 8);
		circle(frame, partA, 8, cv::Scalar(0, 0, 255), -1);
		circle(frame, partB, 8, cv::Scalar(0, 0, 255), -1);
	}

	//cv::imshow("Output-Keypoints", frameCopy);
	//cv::imshow("Output-Skeleton", frame);
}

std::deque<cv::Point> leftTrace;
std::deque<cv::Point> rightTrace;

void detectPoseKeypoints(cv::dnn::Net& net, const cv::Mat &inFrame, cv::Mat &outFrame)
{
	int inHeight = 16*7;
	int inWidth = (16.0 * inHeight) / 9.0;
	float thresh = 0.05;

	int frameWidth = inFrame.cols;
	int frameHeight = inFrame.rows;

	cv::Mat inpBlob = cv::dnn::blobFromImage(inFrame, 1.0 / 255, cv::Size(inWidth, inHeight), cv::Scalar(0, 0, 0), false, false);
	net.setInput(inpBlob);

	cv::Mat output = net.forward();

	int H = output.size[2];
	int W = output.size[3];

	const int nPoints = 25;

	bool centreVisible = false, leftVisible = false, rightVisible = false;

	// find the position of the body parts
	std::vector<cv::Point> points(nPoints);
	for (int n = 0; n < nPoints; n++)
	{
		// Probability map of corresponding body's part.
		cv::Mat probMap(H, W, CV_32F, output.ptr(0, n));

		cv::Point2f p(-1, -1);
		cv::Point maxLoc;
		double prob;
		minMaxLoc(probMap, 0, &prob, 0, &maxLoc);
		if (prob > thresh)
		{
			p = maxLoc;
			p.x *= (float)frameWidth / W;
			p.y *= (float)frameHeight / H;
			
			if (n == 1)
			{
				centreVisible = true;
			}
			else if (n == 4 && maxLoc.y < (7 * H / 8))
			{
				leftVisible = true;
			}
			else if (n == 7 && maxLoc.y < (7 * H / 8))
			{
				rightVisible = true;
			}
		}
		
		points[n] = p;
	}
	int nPairs = sizeof(POSE_PAIRS) / sizeof(POSE_PAIRS[0]);

	for (int n = 0; n < nPairs; n++)
	{
		// lookup 2 connected body/hand parts
		cv::Point2f partA = points[POSE_PAIRS[n][0]];
		cv::Point2f partB = points[POSE_PAIRS[n][1]];

		if (partA.x <= 0 || partA.y <= 0 || partB.x <= 0 || partB.y <= 0)
			continue;

		line(outFrame, partA, partB, cv::Scalar(0, 255, 255), 2);
		circle(outFrame, partA, 2, cv::Scalar(0, 0, 255), -1);
		circle(outFrame, partB, 2, cv::Scalar(0, 0, 255), -1);
	}

	float deltaText = 15;
	float scaleText = 0.35;

	/* Parameter 1: 1/2 hands */
	if (leftVisible && rightVisible)
		cv::putText(outFrame, "Hands: Both", cv::Point(0, deltaText), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
	else if (leftVisible && !rightVisible)
		cv::putText(outFrame, "Hands: Left only", cv::Point(0, deltaText), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
	else if (!leftVisible && rightVisible)
		cv::putText(outFrame, "Hands: Right only", cv::Point(0, deltaText), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
	else
		cv::putText(outFrame, "Hands: No hands detected", cv::Point(0, deltaText), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));

	/* Parameter 2: position */
	if (leftVisible && centreVisible)
	{
		cv::Point centre = points[1];
		cv::Point hand = points[4];

		float angle = std::atan2(-(hand.y - centre.y), hand.x - centre.x) * 180.0 / 3.14159265358979323846;


		float tolerance = 20;

		if (angle < 0) angle += 360;

		if ((0 <= angle && angle < tolerance) || (360 - tolerance <= angle))
			cv::putText(outFrame, "Left hand: next to shoulder (R)", cv::Point(0, deltaText * 2), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (tolerance <= angle && angle < 90)
			cv::putText(outFrame, "Left hand: next to head (R)", cv::Point(0, deltaText * 2), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (90 <= angle && angle < 180 - tolerance)
			cv::putText(outFrame, "Left hand: next to head (L)", cv::Point(0, deltaText * 2), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (180 - tolerance <= angle && angle < 180 + tolerance)
			cv::putText(outFrame, "Left hand: next to shoulder (L)", cv::Point(0, deltaText * 2), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (180 + tolerance <= angle && angle < 270)
			cv::putText(outFrame, "Left hand: below shoulder (L)", cv::Point(0, deltaText * 2), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else
			cv::putText(outFrame, "Left hand: below shoulder (R)", cv::Point(0, deltaText * 2), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
	}
	if (rightVisible && centreVisible)
	{
		cv::Point centre = points[1];
		cv::Point hand = points[7];

		float angle = std::atan2(-(hand.y - centre.y), hand.x - centre.x) * 180.0 / 3.14159265358979323846;

		float tolerance = 20;

		if (angle < 0) angle += 360;

		if ((0 <= angle && angle < tolerance) || (360 - tolerance <= angle))
			cv::putText(outFrame, "Right hand: next to shoulder (R)", cv::Point(0, deltaText * 3), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (tolerance <= angle && angle < 90)
			cv::putText(outFrame, "Right hand: next to head (R)", cv::Point(0, deltaText * 3), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (90 <= angle && angle < 180 - tolerance)
			cv::putText(outFrame, "Right hand: next to head (L)", cv::Point(0, deltaText * 3), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (180 - tolerance <= angle && angle < 180 + tolerance)
			cv::putText(outFrame, "Right hand: next to shoulder (L)", cv::Point(0, deltaText * 3), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (180 + tolerance <= angle && angle < 270)
			cv::putText(outFrame, "Right hand: below shoulder (L)", cv::Point(0, deltaText * 3), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else
			cv::putText(outFrame, "Right hand: below shoulder (R)", cv::Point(0, deltaText * 3), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
	}

	if (leftVisible)
	{
		if (leftTrace.empty())
			for (int i = 0; i < 5; ++i)
				leftTrace.push_back(points[4]);

		leftTrace.pop_front();
		leftTrace.push_back(points[4]);

		cv::line(outFrame, leftTrace.back(), leftTrace.front(), cv::Scalar(255, 255, 0), 2, 4);
		

		cv::Point diff = leftTrace.back() - leftTrace.front();
		float angle = std::atan2(-diff.y, diff.x)* 180.0 / 3.14159265358979323846;
		if (angle < 0) angle += 360;

		if (std::sqrt(diff.dot(diff)) <= (inFrame.rows / 16.0))
			cv::putText(outFrame, "Left hand: Not moving", cv::Point(0, deltaText * 4), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if ((0 <= angle && angle < 45) || (360-45 <= angle))
			cv::putText(outFrame, "Left hand: Moving right", cv::Point(0, deltaText * 4), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (45 <= angle && angle < 135)
			cv::putText(outFrame, "Left hand: Moving up", cv::Point(0, deltaText * 4), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (135 <= angle && angle < 225)
			cv::putText(outFrame, "Left hand: Moving left", cv::Point(0, deltaText * 4), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else
			cv::putText(outFrame, "Left hand: Moving down", cv::Point(0, deltaText * 4), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
	}

	if (rightVisible)
	{
		if (rightTrace.empty())
			for (int i = 0; i < 5; ++i)
				rightTrace.push_back(points[7]);

		rightTrace.pop_front();
		rightTrace.push_back(points[7]);

		cv::line(outFrame, rightTrace.back(), rightTrace.front(), cv::Scalar(255, 255, 0), 2, 4);


		cv::Point diff = rightTrace.back() - rightTrace.front();
		float angle = std::atan2(-diff.y, diff.x)* 180.0 / 3.14159265358979323846;
		if (angle < 0) angle += 360;

		if (std::sqrt(diff.dot(diff)) <= (inFrame.rows / 32.0))
			cv::putText(outFrame, "Right hand: Not moving", cv::Point(0, deltaText * 5), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if ((0 <= angle && angle < 45) || (360 - 45 <= angle))
			cv::putText(outFrame, "Right hand: Moving right", cv::Point(0, deltaText * 5), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (45 <= angle && angle < 135)
			cv::putText(outFrame, "Right hand: Moving up", cv::Point(0, deltaText * 5), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else if (135 <= angle && angle < 225)
			cv::putText(outFrame, "Right hand: Moving left", cv::Point(0, deltaText * 5), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
		else
			cv::putText(outFrame, "Right hand: Moving down", cv::Point(0, deltaText * 5), cv::FONT_HERSHEY_SIMPLEX, scaleText, cv::Scalar(255, 255, 255));
	}
}

int main(int argc, const char** argv)
{
	cv::dnn::Net faceNet = cv::dnn::readNetFromCaffe("./face.prototxt", "./face.caffemodel");
	cv::dnn::Net handNet = cv::dnn::readNetFromCaffe("./hand.prototxt", "./hand.caffemodel");
	cv::dnn::Net poseNet = cv::dnn::readNetFromCaffe("./pose.prototxt", "./pose.caffemodel");

	cv::VideoCapture source("welkomstwoord.mp4");
	cv::Mat frame;

	source >> frame;

	cv::Rect handBoundingBox(0, 0, frame.cols, frame.rows);
	cv::Mat handFrame;

	cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(frame.cols, frame.rows));

	while (true)
	{
		source >> frame;

		if (frame.empty())
			break;

		cv::Mat flipped;
		cv::flip(frame, flipped, 1);

		cv::Mat output = flipped.clone();


		detectPoseKeypoints(poseNet, flipped, output);
		detectFaceOpenCVDNN(faceNet, flipped, output);

		//cv::rectangle(output, handBoundingBox, cv::Scalar(255, 0, 0), 2, 4);

		cv::addWeighted(flipped, 0.6, output, 0.4, 0, output);

		imshow("Input", output);

		int k = cv::waitKey(10);

		if (k == 'h')
		{
			handBoundingBox = cv::selectROI("Input", flipped);
		}

		if (k == 'r')
		{
			handFrame = flipped(handBoundingBox);
			detectHandKeypoints(handNet, handFrame);
			cv::imshow("Hand", handFrame);
		}

		if (k == 27)
		{
			cv::destroyAllWindows();
			break;
		}

		video.write(output);
	}

	return 0;
}
