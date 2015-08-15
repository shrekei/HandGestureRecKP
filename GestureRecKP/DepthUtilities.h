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

#pragma once
#include <vector>
#include <math.h>
#include <cxcore.h>
#include <highgui.h>
#include <cv.h>

// depth feature: a pair of neighbor points
class DepthFeature
{
public:
	DepthFeature(void) {}
	DepthFeature(cv::Point2f  u0) : u(u0){}
	cv::Point2f u;
};

// parameters and functions used to calculate the depth features
extern double	g_f, g_fu, g_fv;
extern int		g_u0, g_v0;
extern double	g_fPlaneDepth;
extern double	g_fMaxDepthDiff;
extern cv::Mat	g_mtxDepthPoints;
extern std::vector<DepthFeature>	g_vecFeatureIndices;

inline void SetFeatureParam(double f0, double fu0, double fv0,
	int u00, int v00, double fPlaneDepth, double fMaxDepthDiff)
{
	g_f = f0;
	g_fu = fu0;
	g_fv = fv0; 
	g_u0 = u00; 
	g_v0 = v00;	
	g_fPlaneDepth = fPlaneDepth;
	g_fMaxDepthDiff = fMaxDepthDiff;
}

inline void GenerateFeatureIndicesApt(vector<DepthFeature> &vecFeatureIndices,
	int nFeatureDim, int nAnchor, double fRadius)
{
	int nGridPointNum = (2 * nAnchor + 1) * (2 * nAnchor + 1) - 1;
	assert(nGridPointNum == nFeatureDim );
	
	// generate the grid points
	double c0, k;
	double fRatio = 0.2;
	c0 = nAnchor / (fRadius + (fRatio - 1) * fRadius / 2);
	k = (fRatio - 1) * c0 / fRadius;
	vecFeatureIndices.resize(nFeatureDim);
	int nFeatCount = 0;
	for (int i = -nAnchor; i <= nAnchor; i++)
	{
		for (int j = -nAnchor; j <= nAnchor; j++)
		{
			double si, sj, vi, vj;
			si = i > 0 ? 1 : -1;
			sj = j > 0 ? 1 : -1;
			vi = fabs(1.0 * i);
			vj = fabs(1.0 * j);
			if (i == 0 && j == 0)
				continue;
			double ux = sj * (-c0 + sqrt(c0 * c0 + 4 * vj * k / 2)) / k;
			double uy = si * (-c0 + sqrt(c0 * c0 + 4 * vi * k / 2)) / k;
			DepthFeature dfTmp(cv::Point2f(-ux, uy));
			vecFeatureIndices.at(nFeatCount++) = dfTmp;
		}
	}
}

inline double CalcFeatureValue(DepthFeature ftIn, cv::Point p, double fDepthValue, 
	double f, double fu, double fv, const cv::Mat &mtxDepthPoints)
{
	if (fDepthValue == 0.0)
		return 0.0;

	cv::Point ud;
	ud.x = ftIn.u.x / fDepthValue * f * fu + p.x;
	ud.y = ftIn.u.y / fDepthValue * f * fv + p.y;
	
	double fDepthU;
	if (ud.x < 0 || ud.x >= mtxDepthPoints.cols || ud.y < 0 || ud.y >= mtxDepthPoints.rows)
		fDepthU = g_fPlaneDepth;						// assumed to be background
	else
		fDepthU = mtxDepthPoints.at<cv::Vec3f>(ud.y, ud.x)[2];
	
	double fFeatureValue = fDepthU - fDepthValue; 
	if (fabs(fFeatureValue) > g_fMaxDepthDiff)
	{
		double fSign = fFeatureValue >= 0 ? 1 : -1;
		fFeatureValue = fSign * g_fMaxDepthDiff;
	}
	return fFeatureValue;
}
