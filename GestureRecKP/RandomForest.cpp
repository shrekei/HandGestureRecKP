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

#include "RandomForest.h"
#include "DepthUtilities.h"
#include <omp.h>

RandomForest::RandomForest(void)
{
	m_bUseSoftVoting = true;
	m_nTreeNum = 0;
}

RandomForest::~RandomForest() 
{
	for (vector<RandomDecisionTree*>::iterator itrdt = m_vecTrees.begin();
		itrdt != m_vecTrees.end(); itrdt++)
		delete (*itrdt);
}

Result RandomForest::eval(cv::Point p)
{
	Result result, treeResult;
	for (int i = 0; i < m_nClassNum; i++) 
		result.Conf.push_back(0.0);

	for (int i = 0; i < m_nTreeNum; i++)
	{
		treeResult = m_vecTrees[i]->eval(p);
		if (m_bUseSoftVoting) 
			add(treeResult.Conf, result.Conf);
		else 
			result.Conf[treeResult.Pred]++;
	}

	scale(result.Conf, 1.0 / m_nTreeNum);
	result.Pred = argmax(result.Conf);
	int confcounter = 0;
	for (vector<double>::iterator itc = result.Conf.begin();
		itc != result.Conf.end(); itc++)
	{
		confcounter++;
	}
	return result;
}

// generate the confidence map using the depth context	
void RandomForest::test(const cv::Mat &mtxDepthPoints, cv::Mat &mtxLabels,
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

void RandomForest::readForest(std::string strFileName)
{
	FILE *pFile = fopen(strFileName.c_str(), "r");
	int nUseSoftVoting, nMaxDepth, nMinCounter, nProjFeatureNum, nRandomTestNum;
	fscanf(pFile, "%d	%d %d	%d	%d	%d	%d	%d\n", &m_nClassNum, &m_nFeatureDim, &m_nTreeNum, 
		&nUseSoftVoting, &nMaxDepth, &nMinCounter, &nProjFeatureNum, &nRandomTestNum);
	m_bUseSoftVoting = (nUseSoftVoting == 0) ? true : false;
	for (int i = 0; i < m_nTreeNum; i++)
	{
		RandomDecisionTree *pTree = new RandomDecisionTree(i);
		pTree->readTree(pFile, m_nClassNum, m_nFeatureDim);
		m_vecTrees.push_back(pTree);
	}
	fclose(pFile);
}