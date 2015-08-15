#include <iostream>
#include "RandomDecisionTree.h"
#include "DepthUtilities.h"
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

#include <omp.h>
using namespace std;

RandomDecisionTree::RandomDecisionTree(int nTreeLabel)
{
	m_nTreeLabel = nTreeLabel;
	m_pRootNode = NULL;
	m_nNodeNum = 0;
}

RandomDecisionTree::~RandomDecisionTree() 
{
	if (m_pRootNode != NULL)
		delete m_pRootNode;
	m_vecNodes.clear();
	m_nNodeNum = 0;
}

Result RandomDecisionTree::eval(cv::Point p) 
{
	Result result = m_pRootNode->eval(p);
	int confcounter = 0;
	for (vector<double>::iterator itc = result.Conf.begin();
		itc != result.Conf.end(); itc++)
	{
		confcounter++;
	}
	return result;
}

void RandomDecisionTree::test(const cv::Mat &mtxDepthPoints, cv::Mat &mtxLabels,
	float mtxPredictions[240][320][14], cv::Rect rtOBB)
{
	g_mtxDepthPoints = mtxDepthPoints;
	vector<Result> results;

#pragma omp parallel for ordered
	for (int i = rtOBB.y; i < rtOBB.y + rtOBB.height; i++)
	{
		for (int j = rtOBB.x; j < rtOBB.x + rtOBB.width; j++)
		{
			if (mtxDepthPoints.at<cv::Vec3f>(i, j)[2] == g_fPlaneDepth)
			{
				mtxLabels.at<int>(i, j) = 13;
				for (int k = 0; k < 14; k++)
					mtxPredictions[i][j][k] = 0.0;
			}
			else
			{
				Result result = eval(cvPoint(j, i));
				int label = result.Pred;
				mtxLabels.at<int>(i, j) = label;
				for (int k = 0; k < 14; k++)
					mtxPredictions[i][j][k] = result.Conf[k];
			}
		}
	}
}

void RandomDecisionTree::readTree(FILE *pFile, int nClassNum, int nFeatureDim)
{
	m_nClassNum = nClassNum;
	m_nFeatureDim = nFeatureDim;
	m_pRootNode = new RDTNode(0, m_nClassNum, m_nFeatureDim);
	m_vecNodes.push_back(m_pRootNode);
	m_pRootNode->readNode(pFile, m_vecNodes);
}