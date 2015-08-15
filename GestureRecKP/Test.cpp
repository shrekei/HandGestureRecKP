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

bool LoadDepthPoints(cv::Mat &mtxDepthPoints, char *strName)
{
	FILE *pFile = fopen(strName, "rb");
	if (!pFile)
		return false;
	int nWidth, nHeight;
	fread(&nWidth, sizeof(int), 1, pFile);
	fread(&nHeight, sizeof(int), 1, pFile);
	if (mtxDepthPoints.cols != nWidth || mtxDepthPoints.rows != nHeight)
	{
		mtxDepthPoints.release();
		mtxDepthPoints.create(cv::Size(nWidth, nHeight), CV_32FC3);
	}
	for (int i = 0; i < mtxDepthPoints.rows; i++)
	{
		for (int j = 0; j < mtxDepthPoints.cols; j++)
		{
			float x, y, z;
			fread(&x, sizeof(float), 1, pFile);
			fread(&y, sizeof(float), 1, pFile);
			fread(&z, sizeof(float), 1, pFile);
			mtxDepthPoints.at<cv::Vec3f>(i, j) = cv::Vec3f(x, y, z);
		}
	}
	fclose(pFile);
	return true;
}

int main(void)
{
	HandProcessor gProcessor;
	gProcessor.Init("..\\data\\Config.txt");

	for (int i = 0; i < 985; i++)
	{
		char text[255];
		sprintf(text, "..\\..\\2014_ReleaseGestureSet\\depth3d_%d.mat", i);
		cv::Mat mtxDepthPoints;
		int nPredGesture;
		if (LoadDepthPoints(mtxDepthPoints, text))
		{
			gProcessor.Update(mtxDepthPoints, nPredGesture);
		}
		cv::waitKey(5);
	}
}