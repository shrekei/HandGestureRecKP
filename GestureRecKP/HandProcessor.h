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
#include "RandomForest.h"
#include "GestureRec.h"

class HandProcessor
{
public:
	void	Init(const string& confFile);
	bool	Update(const cv::Mat &mtxDepthPoints, int& nGesture);
	cv::Mat	DrawPartLabels(const cv::Mat &mtxPartLabels);

private:		// hand region detection
	bool	LocateHand(cv::Mat &mtxDepthPoints, double fMinDepth, double fMaxDepth, cv::Mat &mtxMaskImg, HandROI &gHand);
	void	ThresholdROI(const cv::Mat &mtxDepthPoints, double fMinDepth, double fMaxDepth, cv::Mat &mtxMaskImg, cv::Rect &rtOBB);
	bool	RefineROI(cv::Mat &mtxDepthPoints, cv::Mat &mtxMaskImg, double fMinArea, HandROI &gHand);
	
private:		// hand parsing thread
	void	AssignRFLabels(cv::Mat &mtxDepthRF, cv::Rect rtOBB, cv::Mat &mtxPartLabels, float mtxPredictions[240][320][14], bool bRight);
	void	AssignPartLabels(const cv::Mat &mtxDepthPoints, cv::Mat &mtxPartLabels, HandROI &gHand);

private:		// hand joint extraction
	bool	ExtractJoints(cv::Mat &mtxDepthPoints, cv::Mat &mtxPartLabels, float mtxPredictions[240][320][14], HandROI &gHand);
	bool	InterpolateJoints(HandROI &gHand);
	std::vector<cv::Vec3f>	SeekModes(cv::Mat &mtxDepthPoints, float mtxPredictions[240][320][14], cv::Rect rt);
	int		ClassifyGesture(const HandROI &gHand);

private:	
	cv::Size				m_sFrame;	
	cv::Mat				m_mtxDepthPoints;
	double				m_fMinDepth, m_fMaxDepth;
	RandomForest		m_gRF;
	HandROI			m_gHand;	
	
private:
	static const int		MIN_HAND_SIZE;
	static const int		MIN_PART_SIZE;
};