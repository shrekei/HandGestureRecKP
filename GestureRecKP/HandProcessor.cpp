/*
Copyright (c) 2014, Hui Liang
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to 
endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF 
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Reference
[1] Hui Liang, Junsong Yuan and Daniel Thalmann, Parsing the Hand in Depth Images, in IEEE Trans. Multimedia, Aug. 2014.
[2] Hui Liang and Junsong Yuan, Hand Parsing and Gesture Recognition with a Commodity Depth Camera, 
in Computer Vision and Machine Learning with RGB-D Sensors, Springer, 2014.
*/

#include "HandProcessor.h"
#include <sys/timeb.h>
#include <sstream>
#include <fstream>
#include <iostream>
using namespace std;

float LabelPredictions[240][320][14];
const int HandProcessor::MIN_HAND_SIZE = 1200;
const int HandProcessor::MIN_PART_SIZE = 20;

void HandProcessor::Init(const string& confFile)
{
	// parse the configuration file
	fstream fsConfig(confFile);
	string strLine;
	
	cout << "Parsing the configuration file" << " ... " << endl;
	getline(fsConfig, strLine);
	stringstream(strLine) >> m_sFrame.width >> m_sFrame.height;
	cout << "Size: " << m_sFrame.width << ", " << m_sFrame.height << endl;
	getline(fsConfig, strLine);
	stringstream(strLine) >> m_fMinDepth >> m_fMaxDepth;
	cout << "Depth Range: " << m_fMinDepth << ", " << m_fMaxDepth << endl;

	// read the Random Forest parameters	
	int nAnchor;
	double f, fu, fv, fRadius;
	string strForestFile;
	getline(fsConfig, strLine);
	stringstream(strLine) >> nAnchor;
	cout << "Anchor Number: " << nAnchor << endl;
	getline(fsConfig, strLine);
	stringstream(strLine) >> fRadius;
	cout << "Feature Radius: " << fRadius << endl;
	getline(fsConfig, strLine);
	stringstream(strLine) >> f >> fu >> fv;
	cout << "RF Camera Params: " << f << ", " << fu << ", " << fv << endl;
	getline(fsConfig, strLine);
	int nLen = 1 + min(strLine.find_last_not_of('\t', strLine.find("//") - 1), strLine.find_last_not_of('\0x20', strLine.find("//") - 1));
	strForestFile = strLine.substr(0,  nLen);
	cout << "RF File Name: " << strForestFile << endl;
	
	// set the parameters for the members
	int nFeatureDim = (2 * nAnchor + 1) * (2 * nAnchor + 1) - 1;
	SetFeatureParam(f, fu, fv, m_sFrame.width / 2, m_sFrame.height / 2, 2.0, 0.3);
	GenerateFeatureIndicesApt(g_vecFeatureIndices, nFeatureDim, nAnchor, fRadius);
	m_gRF.readForest(strForestFile);
}

bool HandProcessor::Update(const cv::Mat &mtxDepthPoints, int& nGesture)
{	
	static cv::Mat mtxPartLabels(mtxDepthPoints.size(), CV_32SC1);
	static cv::Mat mtxMask(mtxDepthPoints.size(), CV_8UC1);
	static cv::Mat mtxDepthDly(mtxDepthPoints.size(), CV_8UC1);
	for (int i = 0; i < mtxDepthPoints.rows; i++)
	{
		for (int j = 0; j < mtxDepthPoints.cols; j++)
			if (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] >= m_fMinDepth && mtxDepthPoints.at<cv::Vec3f>(i, j)[2] <= m_fMaxDepth)
				mtxDepthDly.at<uchar>(i, j) = (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] - m_fMinDepth) / (m_fMaxDepth - m_fMinDepth) * 255;
			else
				mtxDepthDly.at<uchar>(i, j) = 0;
	}
		
	mtxDepthPoints.copyTo(m_mtxDepthPoints);
	if (!LocateHand(m_mtxDepthPoints, m_fMinDepth, m_fMaxDepth, mtxMask, m_gHand))
	{
		nGesture = -1;
		return false;	
	}
	
	for (int i = 0; i < m_mtxDepthPoints.rows; i++)
	{
		for (int j = 0; j < m_mtxDepthPoints.cols; j++)
			if (m_mtxDepthPoints.at<cv::Vec3f>(i, j)[2] == 0.0)
				m_mtxDepthPoints.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, g_fPlaneDepth);
	}
	AssignPartLabels(m_mtxDepthPoints, mtxPartLabels, m_gHand);
	if (!ExtractJoints(m_mtxDepthPoints, mtxPartLabels, LabelPredictions, m_gHand))
	{
		nGesture = -1;
		return false;
	}

	cv::Mat mtxPartColor = DrawPartLabels(mtxPartLabels);

	nGesture = ClassifyGesture(m_gHand);
	cout << nGesture << endl;
	
	cv::Size sDepth = mtxDepthPoints.size();
	static cv::Mat mtxUI(sDepth * 2, CV_8UC3);
	static cv::Mat mtxTemp(sDepth, CV_8UC3);
	cv::cvtColor(mtxDepthDly, mtxTemp, CV_GRAY2BGR);
	mtxTemp.copyTo(mtxUI(cv::Rect(sDepth.width, 0, sDepth.width, sDepth.height)));
	cv::cvtColor(mtxMask, mtxTemp, CV_GRAY2BGR);
	mtxTemp.copyTo(mtxUI(cv::Rect(0, 0, sDepth.width, sDepth.height)));
	mtxPartColor.copyTo(mtxUI(cv::Rect(0, sDepth.height, sDepth.width, sDepth.height)));
	mtxTemp.setTo(cv::Scalar(0, 0, 0));
	char text[255];
	sprintf(text, "Recognition results: %d", nGesture);
	cv::putText(mtxTemp, text, cv::Point(5, 110), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
	mtxTemp.copyTo(mtxUI(cv::Rect(sDepth.width, sDepth.height, sDepth.width, sDepth.height)));
	cv::imshow("Test", mtxUI);
	return true;
}

vector<cv::Vec3f> HandProcessor::SeekModes(cv::Mat &mtxDepthPoints, float mtxPredictions[240][320][14], cv::Rect rt)
{
	// estimate the prior P(l) = sum(P(l|v)P(v))
	double pl[11] = {0};
	for (int k = 1; k < 13; k++)
	{
		int nIndex = (k == 1 || k == 2) ? 0 : k - 2;
		for (int i = rt.y; i < rt.y + rt.height; i++)
		{
			for (int j = rt.x; j < rt.x + rt.width; j++)
				pl[nIndex] += mtxPredictions[i][j][k];
		}
	}

	// estimate the initial centroid of each part
	vector<cv::Vec3f> vecJoints(11, cv::Vec3f(0, 0, 0));
	for (int k = 1; k < 13; k++)
	{
		int nIndex = (k == 1 || k == 2) ? 0 : k - 2;
		double fInvSum = 1.0 / pl[nIndex];
		for (int i = rt.y; i < rt.y + rt.height; i++)
		{
			for (int j = rt.x; j < rt.x + rt.width; j++)
			{
				vecJoints[nIndex][0] += fInvSum * mtxPredictions[i][j][k] * mtxDepthPoints.at<cv::Vec3f>(i, j)[0];
				vecJoints[nIndex][1] += fInvSum * mtxPredictions[i][j][k] * mtxDepthPoints.at<cv::Vec3f>(i, j)[1];
				vecJoints[nIndex][2] += fInvSum * mtxPredictions[i][j][k] * mtxDepthPoints.at<cv::Vec3f>(i, j)[2];
			}
		}
	}

	// perform mean-shift to seek the local modes
	cv::Vec3f dev(0.04, 0.04, 0.04);
	for (int nIter = 0; nIter < 6; nIter++)
	{
		double den[11] = {0};
		vector<cv::Vec3f> vecNew(11, cv::Vec3f(0, 0, 0));
		for (int k = 1; k < 13; k++)
		{
			int nIndex = (k == 1 || k == 2) ? 0 : k - 2;
			double fInvSum = 1.0 / pl[nIndex];
			cv::Vec3f vOrg = vecJoints[nIndex];
			for (int i = rt.y; i < rt.y + rt.height; i++)
			{
				for (int j = rt.x; j < rt.x + rt.width; j++)
				{
					cv::Vec3f vCur = mtxDepthPoints.at<cv::Vec3f>(i, j);
					double kernel = exp(-0.5*(((vCur[0] - vOrg[0]) / dev[0]) * ((vCur[0] - vOrg[0]) / dev[0]) +
						((vCur[1] - vOrg[1]) / dev[1]) * ((vCur[1] - vOrg[1]) / dev[1]) + 
						((vCur[2] - vOrg[2]) / dev[2]) * ((vCur[2] - vOrg[2]) / dev[2])));
					vecNew[nIndex][0] += vCur[0] * fInvSum * mtxPredictions[i][j][k] * kernel;
					vecNew[nIndex][1] += vCur[1] * fInvSum * mtxPredictions[i][j][k] * kernel;
					vecNew[nIndex][2] += vCur[2] * fInvSum * mtxPredictions[i][j][k] * kernel;
					den[nIndex] += fInvSum * mtxPredictions[i][j][k] * kernel;
				}
			}
		}
		for (int k = 0; k < 11; k++)
		{
			if (den[k] != 0)
			{
				vecJoints[k][0] = vecNew[k][0] / den[k];
				vecJoints[k][1] = vecNew[k][1] / den[k];
				vecJoints[k][2] = vecNew[k][2] / den[k];
			}
		}
	}
	return vecJoints;
}

bool HandProcessor::ExtractJoints(cv::Mat &mtxDepthPoints, cv::Mat &mtxPartLabels, 
	float mtxPredictions[240][320][14], HandROI &gHand)
{
	cv::Rect rt = gHand.OBB;

	// get the joint positions
	vector<cv::Vec3f> vecJoints = SeekModes(mtxDepthPoints, mtxPredictions, rt);
	gHand.Joints = vecJoints;

	// collect the points within each part
	std::vector<cv::Vec3f> vecPartPoints[11];
	for (int i = rt.y; i < rt.y + rt.height; i++)
	{
		for (int j = rt.x; j < rt.x + rt.width; j++)
		{
			int nLabel = mtxPartLabels.at<int>(i, j);
			if (nLabel != 0 && nLabel != 13)
			{
				cv::Vec3f vCur = mtxDepthPoints.at<cv::Vec3f>(i, j);
				int nIndex = (nLabel == 1 || nLabel == 2) ? 0 : nLabel - 2;
				if (nIndex == 0)
					vecPartPoints[nIndex].push_back(vCur);
				else
				{
					// eliminate the outliers
					if (cv::norm(vecJoints[nIndex] - vCur) < 0.04)
						vecPartPoints[nIndex].push_back(vCur);
				}
			}
		}
	}
	for (int i = 0; i < 11; i++)
	{
		if (vecPartPoints[i].size() < MIN_PART_SIZE)
			gHand.Active[i] = false;
		else
		{
			gHand.Active[i] = true;
			gHand.Sizes[i] = vecPartPoints[i].size();
		}
	}
	if (!InterpolateJoints(gHand))
		return false;
	else
		return true;
}

bool HandProcessor::InterpolateJoints(HandROI &gHand)
{
	if (gHand.Joints.size() == 11)
	{
		// do not proceed if the palm is not detected
		if (!gHand.Active[0])
			return false;

		// interpolate the missing hand parts
		for (int i = 1; i < 11; i++)
		{
			// only one case is handled for the missing parts that have two available neighbors
			if (!gHand.Active[i])
			{
				int prev = Neighbors[0][i];
				int next = Neighbors[1][i];
				if ((prev != -1 && next != -1) && 
					gHand.Active[prev] && gHand.Active[next])
				{
					gHand.Joints[i][0] = 0.5 * (gHand.Joints[prev][0] + gHand.Joints[next][0]);
					gHand.Joints[i][1] = 0.5 * (gHand.Joints[prev][1] + gHand.Joints[next][1]);
					gHand.Joints[i][2] = 0.5 * (gHand.Joints[prev][2] + gHand.Joints[next][2]);
				}
			}
			else
			{
				int prev = Neighbors[0][i];
				if (prev != -1 && !gHand.Active[prev])
					gHand.Active[i] = false;
			}
		}
		return true;
	}
	return false;
}

bool HandProcessor::LocateHand(cv::Mat &mtxDepthPoints, double fMinDepth, double fMaxDepth,
	cv::Mat &mtxMaskImg, HandROI &gHand)
{
	cv::Rect rtOBB;
	ThresholdROI(mtxDepthPoints, fMinDepth, fMaxDepth, mtxMaskImg, rtOBB);
	bool bFound = RefineROI(mtxDepthPoints, mtxMaskImg, MIN_HAND_SIZE, gHand);
	return bFound;
}

void HandProcessor::ThresholdROI(const cv::Mat &mtxDepthPoints, double fMinDepth, double fMaxDepth, 
	cv::Mat &mtxMaskImg, cv::Rect &rtOBB)
{
	int xmin, xmax, ymin, ymax;
	xmin = ymin = 1e8;
	xmax = ymax = 0;

	for (int i = 0; i < mtxDepthPoints.rows; i++)
	{
		for (int j = 0; j < mtxDepthPoints.cols; j++)
		{
			double fDepth = mtxDepthPoints.at<cv::Vec3f>(i, j)[2];
			if (fDepth < fMinDepth || fDepth > fMaxDepth)
				mtxMaskImg.at<unsigned char>(i, j) = 0;
			else
			{
				mtxMaskImg.at<unsigned char>(i, j) = 255;
				xmin = j < xmin ? j : xmin;
				xmax = j > xmax ? j : xmax;
				ymin = i < ymin ? i : ymin;
				ymax = i > ymax ? i : ymax;
			}
		}
	}
	rtOBB = cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

bool HandProcessor::RefineROI(cv::Mat &mtxDepthPoints, cv::Mat &mtxMaskImg, double fMinArea, HandROI &gHand)
{
	// perform contour detection with OPENCV
	cv::Mat mtxTmp;
	mtxMaskImg.copyTo(mtxTmp);
	vector<vector<cv::Point> > vecContours;
	cv::findContours(mtxTmp, vecContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	if (vecContours.size() == 0)
	{
		cout << "No contour is found!" << endl;
		return false;
	}

	bool bFound = false;
	for (vector<vector<cv::Point> >::iterator itc = vecContours.begin(); 
		itc != vecContours.end(); itc++) 
	{
		double fArea = fabs(cv::contourArea(cv::Mat(*itc)));
		if ( fArea > fMinArea)
		{
			// check whether the reference center is in the bounding rectangle 
			cv::Rect rtOBB = cv::boundingRect(cv::Mat(*itc));
			gHand.OBB = rtOBB;
			bFound = true;
			break;
		}
	}

	// refine the mask image
	for (int i = 0; i < mtxMaskImg.rows; i++)
	{
		for (int j = 0; j < mtxMaskImg.cols; j++)
		{
			if (mtxMaskImg.at<unsigned char>(i, j) == 0)
				mtxDepthPoints.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
			else
			{
				bool bFlag = false;
				if (cv::Point(j, i).inside(gHand.OBB))
				{
					bFlag = true;
					break;
				}
				if (!bFlag)
				{
					mtxMaskImg.at<unsigned char>(i, j) = 0;
					mtxDepthPoints.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
				}
			}
		}
	}
	return true;
}

void HandProcessor::AssignPartLabels(const cv::Mat &mtxDepthPoints, cv::Mat &mtxPartLabels, HandROI &gHand)
{	
	for (int i = 0; i < mtxDepthPoints.rows; i++)
	{
		for (int j = 0; j < mtxDepthPoints.cols; j++)
		{
			// initialize hand part classification
			mtxPartLabels.at<int>(i, j) = 13;
			for (int k = 0; k < 14; k++)
				LabelPredictions[i][j][k] = 0.0;
		}
	}
	g_mtxDepthPoints = mtxDepthPoints;

	cv::Rect rtOBB = gHand.OBB;
// #pragma omp parallel for ordered
	for (int i = rtOBB.y; i < rtOBB.y + rtOBB.height; i++)
	{
		for (int j = rtOBB.x; j < rtOBB.x + rtOBB.width; j++)
		{
			int x = j;
			int y = i;
			if (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] == g_fPlaneDepth)
			{
				mtxPartLabels.at<int>(y, x) = 13;
				for (int k = 0; k < 14; k++)
					LabelPredictions[y][x][k] = 0.0;
			}
			else
			{
				Result result = m_gRF.eval(cvPoint(j, i));
				int label = result.Pred;
				mtxPartLabels.at<int>(y, x) = label;
				for (int k = 0; k < 14; k++)
					LabelPredictions[y][x][k] = result.Conf[k];
			}
		}
	}
}

cv::Mat HandProcessor::DrawPartLabels(const cv::Mat &mtxPartLabels)
{
	const int COLORS[14] = {	0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF, 
		0x800000, 	0x008000, 0x000080, 0x808000, 0x800080, 0x008080, 0x808080, 0xC00000};

	static cv::Mat mtxPartColor(mtxPartLabels.size(), CV_8UC3);
	mtxPartColor.setTo(cv::Scalar(0, 0, 0));
	unsigned char R, G, B;
	for (int i = 0; i < mtxPartLabels.rows; i++)
	{
		for (int j = 0; j < mtxPartLabels.cols; j++)
		{
			if (mtxPartLabels.at<int>(i, j) != 13)
			{
				int nValue = COLORS[mtxPartLabels.at<int>(i, j)];
				R = (nValue & 0xFF0000) >> 16;
				G = (nValue & 0x00FF00) >> 8;
				B = nValue & 0x0000FF;
				mtxPartColor.at<cv::Vec3b>(i, j) = cv::Vec3b(B, G, R);
			}
		}
	}
	return mtxPartColor;
}

int HandProcessor::ClassifyGesture(const HandROI &gHand)
{
	vector<cv::Vec3f> vJoints = gHand.Joints;
	cv::Vec3f vDataCenter(0, 0, 0);
	for (int j = 0; j < 11; j++)
	{
		if (!gHand.Active[j])
			continue;

		vDataCenter[0] += vJoints[j][0];
		vDataCenter[1] += vJoints[j][1];
		vDataCenter[2] += vJoints[j][2];
	}
	for (int j = 0; j < 11; j++)
	{
		vJoints[j][0] -= vDataCenter[0] / 11.0;
		vJoints[j][1] -= vDataCenter[1] / 11.0;
		vJoints[j][2] -= vDataCenter[2] / 11.0;
	}

	// shift the center of the joints
	double fMinDist = 1.0e10;
	int nLabel;
	double fDists[10];
	for (int i = 0; i < 10; i++)
	{
		// get the joint positions of the candidate
		const double (*const pTemplate)[3][11] = TEMPLATES[i];
		vector<cv::Vec3f> vCand(11);
		cv::Vec3f vCandCenter = cv::Vec3f(0, 0, 0);
		for (int j = 0; j < 11; j++)
		{
			vCand[j][0] = (*pTemplate)[0][j];
			vCand[j][1] = (*pTemplate)[1][j];
			vCand[j][2] = (*pTemplate)[2][j];
			vCandCenter[0] += vCand[j][0];
			vCandCenter[1] += vCand[j][1];
			vCandCenter[2] += vCand[j][2];
		}
		for (int j = 0; j < 11; j++)
		{
			vCand[j][0] -= vCandCenter[0] / 11.0;
			vCand[j][1] -= vCandCenter[1] / 11.0;
			vCand[j][2] -= vCandCenter[2] / 11.0;
		}

		// get the covariance of the input and the candidate
		cv::Mat mtxCov = cv::Mat(3, 3, CV_32FC1);
		mtxCov.setTo(cv::Scalar(0, 0, 0));
		for (int j = 0; j < 11; j++)
		{
			double xi = vJoints[j][0];
			double yi = vJoints[j][1];
			double zi = vJoints[j][2];
			double xt = vCand[j][0];
			double yt = vCand[j][1];
			double zt = vCand[j][2];
			mtxCov.at<float>(0, 0) += xi * xt;
			mtxCov.at<float>(0, 1) += xi * yt;
			mtxCov.at<float>(0, 2) += xi * zt;
			mtxCov.at<float>(1, 0) += yi * xt;
			mtxCov.at<float>(1, 1) += yi * yt;
			mtxCov.at<float>(1, 2) += yi * zt;
			mtxCov.at<float>(2, 0) += zi * xt;
			mtxCov.at<float>(2, 1) += zi * yt;
			mtxCov.at<float>(2, 2) += zi * zt;
		}

		// perform SVD decomposition
		cv::SVD svdCov(mtxCov);
		cv::Mat R = svdCov.vt.t() * svdCov.u.t();

		// correct the rotation of the input data
		cv::Vec3f vNewCenter(0, 0, 0);
		cv::Vec3f vNewJoints[11];
		vCandCenter = cv::Vec3f(0, 0, 0);
		for (int j = 0; j < 11; j++)
		{
			cv::Mat mtxData(3, 1, CV_32FC1);
			mtxData.at<float>(0, 0) = vJoints[j][0];
			mtxData.at<float>(1, 0) = vJoints[j][1];
			mtxData.at<float>(2, 0) = vJoints[j][2];
			mtxData = R * mtxData;
			vNewJoints[j] = cv::Vec3f(mtxData.at<float>(0, 0), mtxData.at<float>(1, 0), mtxData.at<float>(2, 0));		

			// find the centers of the transformed input and template
			vNewCenter[0] += vNewJoints[j][0];	
			vNewCenter[1] += vNewJoints[j][1];	
			vNewCenter[2] += vNewJoints[j][2];
			vCandCenter[0] += vCand[j][0];
			vCandCenter[1] += vCand[j][1];
			vCandCenter[2] += vCand[j][2];
		}

		// estimate the optimal distance
		double fDist = 0;
		cv::Vec3f vCenterOffset((vCandCenter[0] - vNewCenter[0]) / 11.0, 
			(vCandCenter[1] - vNewCenter[1]) / 11.0, (vCandCenter[2] - vNewCenter[2]) / 11.0);
		for (int j = 0; j < 11; j++)
		{
			vNewJoints[j][0] += vCenterOffset[0];
			vNewJoints[j][1] += vCenterOffset[1];
			vNewJoints[j][2] += vCenterOffset[2];
			fDist += pow((vNewJoints[j][0] - vCand[j][0]), 2) + pow((vNewJoints[j][1] - vCand[j][1]), 2) + pow((vNewJoints[j][2] - vCand[j][2]), 2);
		}

		fDists[i] = fDist;
		if (fMinDist > fDist)
		{
			fMinDist = fDist;
			nLabel = i;
		}
	}

	cout << "Dists to templates: ";
	for (int i = 0; i < 10; i++)
		cout << fDists[i] << " ";
	cout << endl;
	return nLabel;
}