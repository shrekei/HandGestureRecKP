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
#include <cv.h>
#include <cxcore.h>
#include "DepthUtilities.h"

// in this class, the hypothesis is an arbitrary hyperplane
class HyperPlaneTest
{
public:
	HyperPlaneTest(void) {}
	inline bool	eval(cv::Point p);
	inline void	setInfo(double fThreshold, int nProjFeatureNum, std::vector<int> vecProjFeatures, std::vector<double> vecProjWeights);

public:
	int						m_nClassNum;
	double					m_fThreshold;
	double					m_fFeatMin, m_fFeatMax;
	int						m_nProjFeatureNum;
	std::vector<int>			m_vecProjFeatures;
	std::vector<double>	m_vecProjWeights;
};

inline bool HyperPlaneTest::eval(cv::Point p)
{
	double proj = 0.0;
	for (int i = 0; i < m_nProjFeatureNum; i++) 
	{
		cv::Point2f u = g_vecFeatureIndices.at(m_vecProjFeatures[i]).u;
		DepthFeature dfTmp(u);
		double fDepthValue = g_mtxDepthPoints.at<cv::Vec3f>(p.y, p.x)[2];
		double fFeatureValue = CalcFeatureValue(dfTmp, p, fDepthValue, g_f, g_fu, g_fv, g_mtxDepthPoints);
		proj += fFeatureValue * m_vecProjWeights[i];
	}
	return (proj > m_fThreshold) ? true : false;
}

inline void HyperPlaneTest::setInfo(double fThreshold, int nProjFeatureNum,
	std::vector<int> vecProjFeatures, std::vector<double> vecProjWeights)
{
	m_fThreshold = fThreshold;
	m_nProjFeatureNum = nProjFeatureNum;
	m_vecProjFeatures = vecProjFeatures;
	m_vecProjWeights = vecProjWeights;
}

inline int argmax(const vector<double> &inVect) 
{
	double maxValue = inVect[0];
	int maxIndex = 0, i = 1;
	vector<double>::const_iterator itr(inVect.begin() + 1), end(inVect.end());
	while (itr != end) 
	{
		if (*itr > maxValue) 
		{
			maxValue = *itr;
			maxIndex = i;
		}
		++i;
		++itr;
	}
	return maxIndex;
}

inline double sum(const vector<double> &inVect)
{
	double val = 0.0;
	vector<double>::const_iterator itr(inVect.begin()), end(inVect.end());
	while (itr != end)
	{
		val += *itr;
		++itr;
	}
	return val;
}

inline void add(std::vector<double> lv, std::vector<double> &rv)
{
	assert(lv.size() == rv.size());
	int len = lv.size();
	for (int i = 0; i < len; i++)
	{
		rv.at(i) += lv.at(i);
	}
}

inline void scale(std::vector<double> &lv, double ratio)
{
	int len = lv.size();
	for (int i = 0; i < len; i++)
	{
		lv.at(i) *= ratio;
	}
}