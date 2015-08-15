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
#include <iostream>
#include <string>
#include <cv.h>
#include <cxcore.h>

const double GT_0[3][11] = {0.0009,   0.0453,    0.0266,    0.0246 ,   0.0220   , 0.0024 ,   0.0024  , -0.0188  , -0.0141  , -0.0390  , -0.0329,
	-0.0268,   -0.0042 ,   0.0051  ,  0.0381,    0.0176 ,   0.0392 ,   0.0118  ,  0.0354,    0.0111   , 0.0310,    0.0140,
	0.6306 ,   0.6288,    0.6208,    0.6126 ,   0.6104  ,  0.6111  ,  0.6090 ,   0.6134 ,   0.6117  ,  0.6196 ,   0.6193};
const double GT_1[3][11] = {0.0021  ,  0.0453  ,  0.0258  ,  0.0282,    0.0343  ,  0.0024  ,  0.0024 ,  -0.0188  , -0.0141,   -0.0390 ,  -0.0329,
	-0.0240 ,  -0.0042  ,  0.0063  ,  0.0521 ,   0.0883   , 0.0392 ,   0.0118  ,  0.0354 ,   0.0111   , 0.0310  ,  0.0140,
	0.6307  ,  0.6288   , 0.6211 ,   0.6347 ,   0.6349 ,   0.6110  ,  0.6090 ,   0.6134 ,   0.6117   , 0.6196  ,  0.6193};
const double GT_2[3][11] = {0.0021 ,   0.0453   , 0.0254   , 0.0281  ,  0.0343 ,   0.0024    ,0.0019 ,  -0.0188  , -0.0137 ,  -0.0390 ,  -0.0329,
	-0.0194,   -0.0042  ,  0.0063   , 0.0520  ,  0.0883  ,  0.0612,    0.1021 ,   0.0354  ,  0.0110 ,   0.0310   , 0.0140,
	0.6309 ,   0.6288  ,  0.6211   , 0.6347  ,  0.6349  ,  0.6334  ,  0.6336  ,  0.6135 ,   0.6120   , 0.6196  ,  0.6193};
const double GT_3[3][11] = {0.0054,    0.0416 ,   0.0588 ,   0.0058 ,   0.0078  , -0.0122  , -0.0168  , -0.0155  ,-0.0225 ,  -0.0221,   -0.0302,
	-0.0787 ,  -0.0469,   -0.0309  ,  0.0144  ,  0.0596  ,  0.0208 ,   0.0586  , -0.0076,   0.0334  , -0.0218 ,  -0.0008,
	0.5969  ,  0.6091,    0.6014 ,   0.6170  ,  0.6093 ,   0.6102 ,   0.6001 ,   0.5973 ,  0.5979 ,   0.5814 ,   0.5794};
const double GT_4[3][11] = {-0.0013,    0.0453    ,0.0254    ,0.0281   , 0.0343   , 0.0023,    0.0019  , -0.0223 ,  -0.0291 ,  -0.0430 ,  -0.0535,
	-0.0147  , -0.0042    ,0.0063  ,  0.0520  ,  0.0883 ,   0.0612  ,  0.1021,    0.0521    ,0.0901 ,   0.0399 ,   0.0665,
	0.6310 ,   0.6288  ,  0.6211  ,  0.6347 ,   0.6349 ,   0.6334  ,  0.6336    ,0.6338 ,   0.6345  ,  0.6356  ,  0.6352};
const double GT_5[3][11] = {0.0006   , 0.0531   , 0.0646  ,  0.0281  ,  0.0343  ,  0.0023  ,  0.0019 ,  -0.0223  , -0.0291  , -0.0430   ,-0.0535,
	-0.0136,   -0.0023 ,   0.0204    ,0.0520 ,   0.0883 ,   0.0612 ,   0.1021  ,  0.0521 ,   0.0901 ,   0.0399 ,   0.0665,
	0.6312  ,  0.6309 ,   0.6283   , 0.6347,    0.6349  ,  0.6334  ,  0.6336   , 0.6338   , 0.6345 ,   0.6356   , 0.6352};
const double GT_6[3][11] = {0.0004  ,  0.0531   , 0.0646  ,  0.0246,    0.0218  ,  0.0024  ,  0.0024 ,  -0.0189  , -0.0141 ,  -0.0429 ,  -0.0535,
	-0.0225 ,  -0.0023 ,   0.0204   , 0.0381   , 0.0163  ,  0.0392 ,   0.0118   , 0.0355  ,  0.0111 ,   0.0398 ,   0.0665,
	0.6308,    0.6309 ,   0.6283  ,  0.6126   , 0.6112    ,0.6111  ,  0.6090,    0.6134   , 0.6117    ,0.6357  ,  0.6352};
const double GT_7[3][11] = {0.0019    ,0.0531 ,   0.0646   , 0.0282    ,0.0343   , 0.0024  ,  0.0024,   -0.0189 ,  -0.0141 ,  -0.0429  , -0.0535,
	-0.0196 ,  -0.0023  ,  0.0204  ,  0.0521 ,   0.0883   , 0.0392,    0.0118  ,  0.0355   , 0.0111 ,   0.0398 ,   0.0665,
	0.6309 ,   0.6309 ,   0.6283,    0.6347  ,  0.6349  ,  0.6110 ,   0.6090   , 0.6134   , 0.6117 ,   0.6357 ,   0.6352};
const double GT_8[3][11] = {0.0042  ,  0.0531  ,  0.0646   , 0.0282  ,  0.0343,    0.0024  ,  0.0024 ,  -0.0188 ,  -0.0141 ,  -0.0390,   -0.0329,
	-0.0218 ,  -0.0023  ,  0.0204,    0.0521   , 0.0883  ,  0.0392    ,0.0118 ,   0.0354,    0.0111  ,  0.0310  ,  0.0140,
	0.6309  ,  0.6309  ,  0.6283    ,0.6347   , 0.6349 ,   0.6110  ,  0.6090 ,   0.6134   , 0.6117 ,   0.6196 ,   0.6193};
const double GT_9[3][11] = {0.0028,    0.0531 ,   0.0646  ,  0.0246 ,   0.0218  ,  0.0024   , 0.0024  , -0.0188 ,  -0.0141 ,  -0.0390   ,-0.0329,
	-0.0251,   -0.0023  ,  0.0204   , 0.0381    ,0.0163   , 0.0392    ,0.0118,    0.0354 ,  0.0111,    0.0310,    0.0140,
	0.6308 ,   0.6309  ,  0.6283 ,   0.6126,    0.6112  ,  0.6111  ,  0.6090  ,  0.6134  ,  0.6117  ,  0.6196   , 0.6193};
const double (*const TEMPLATES[10])[3][11] = {&GT_0, &GT_1, &GT_2, &GT_3, &GT_4, &GT_5, &GT_6, &GT_7, &GT_8, &GT_9};

const int PAIRS[2][10] = {0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
const int Neighbors[2][11] = {-1, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, -1, 2, -1, 4, -1, 6, -1, 8, -1, 10, -1};

class HandROI
{
public:
	std::vector<cv::Vec3f>	Joints;
	cv::Rect				OBB;
	bool					Active[11];
	int						Sizes[11];
};