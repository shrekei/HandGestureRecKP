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

#include <iostream>
#include <map>
#include "RDTNode.h"
using namespace std;

RDTNode::RDTNode(void)
{
	m_bRoot = true;
	m_pLeftChildNode = m_pRightChildNode = NULL;
	m_nNodeLabel = -1;
}

RDTNode::RDTNode(int nDepth, int nClassNum, int nFeatureDim) : m_nDepth(nDepth),
	m_nClassNum(nClassNum), m_nFeatureDim(nFeatureDim), m_bLeaf(true), 
	m_fCounter(0.0), m_nLabel(-1)
{
	if (m_nDepth == 0)
		m_bRoot = true;
	else
		m_bRoot = false;
	for (int i = 0; i < m_nClassNum; i++)
		m_vecLabelStats.push_back(0.0);
	m_pLeftChildNode = m_pRightChildNode = NULL;
}

RDTNode::~RDTNode()
{
	if (!m_bLeaf)
	{
		delete m_pLeftChildNode;
		delete m_pRightChildNode;
		m_pLeftChildNode = m_pRightChildNode = NULL;
	}
}

// evaluate using the depth context
Result RDTNode::eval(cv::Point p) 
{
	if (m_bLeaf) 
	{
		Result result;
		if (m_fCounter != 0.0)
		{
			result.Conf = m_vecLabelStats;
			scale(result.Conf, 1.0 / m_fCounter);
			result.Pred = m_nLabel;
		}
		else
		{
			for (int i = 0; i < m_nClassNum; i++)
			{
				result.Conf.push_back(1.0 / m_nClassNum);
			}
			result.Pred = 0;
		}
		return result;
	}
	else 
	{
		bool bFlag = m_tOpt.eval(p);
		if (bFlag)
			return m_pRightChildNode->eval(p);
		else 
			return m_pLeftChildNode->eval(p);
	}
}

void RDTNode::readNode(FILE *pFile, vector<RDTNode*> &vecNodes)
{
	int nLeaf;
	fscanf(pFile, "%d	", &nLeaf);
	if (nLeaf == 1)
	{	
		m_bLeaf = false;
		fscanf(pFile, "%d	", &m_nLabel);

		double fThreshold;
		int nProjNum;
		std::vector<int> vecProjFeatures;
		std::vector<double> vecProjWeights;
		fscanf(pFile, "%lf	%d	", &fThreshold, &nProjNum);
		for (int i = 0; i < nProjNum; i++)
		{
			int nProjFeature;
			double fWeight;
			fscanf(pFile, "%d	%lf	", &nProjFeature, &fWeight);
			vecProjFeatures.push_back(nProjFeature);
			vecProjWeights.push_back(fWeight);
		}
		m_tOpt.setInfo(fThreshold, nProjNum, vecProjFeatures, vecProjWeights);

		m_pRightChildNode = new RDTNode(m_nDepth + 1, m_nClassNum, m_nFeatureDim);
		m_pLeftChildNode = new RDTNode(m_nDepth + 1, m_nClassNum, m_nFeatureDim);
		vecNodes.push_back(m_pRightChildNode);
		vecNodes.push_back(m_pLeftChildNode);
		m_pLeftChildNode->readNode(pFile, vecNodes);
		m_pRightChildNode->readNode(pFile, vecNodes);
	}
	else
	{
		m_bLeaf = true;
		fscanf(pFile, "%d	", &m_nLabel);
		vector<double> vecLabelStats;
		for (int i = 0; i < m_nClassNum; i++)
		{
			double fStat;
			fscanf(pFile, "%lf	", &fStat);
			vecLabelStats.push_back(fStat);
		}
		m_vecLabelStats = vecLabelStats;
		m_fCounter = sum(m_vecLabelStats);
		m_pLeftChildNode = m_pRightChildNode = NULL;
	}
}