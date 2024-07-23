#pragma once  //确保头文件只被编译一次

#ifndef LOCALWRAPPPING_h
#define LOCALWRAPPPING_h

#include <vector>
#include <opencv2/core.hpp>
#include "config.h"

using std::vector;
using cv::Mat;

#define top_border 0
#define bottom_border 1
#define left_border 2
#define right_border 3

#define seam_horizontal 0
#define seam_vertical 1

class LocalWrap{
public:

    // 计算像素能量(边缘检测)
    Mat get_EnergyMap(const Mat& src);
    // 找到最长确实边界，确定子图区域
    vector<int> find_Longest_MissingBorder(const Mat& src, const Mat& mask, int& border_type);
    // 根据像素能量获取seampath
    vector<int> get_Seam_Path_Vertical(Mat src, Mat mask, int seam_type, vector<int> begin_end);
    vector<int> get_Seam_Path_Horizontal(Mat src, Mat mask, int seam_type, vector<int> begin_end);
    // 根据seampath移动像素
    void shiftPixels_BySeamPath_Vertical(Mat& src, Mat& mask, vector<int> seamPath, int seam_type, bool shift2end, vector<int> begin_end,  Mat& src_SeamPaths);
    void shiftPixels_BySeamPath_Horizontal(Mat& src, Mat& mask, vector<int> seamPath, int seam_type, bool shift2end, vector<int> begin_end, Mat& src_SeamPaths);
    // 更新位移场
    void update_displacementField(vector<vector<CoordinateInt>>& displacementField, vector<vector<CoordinateInt>>& Final_displacementField, 
        int seam_type, vector<int> begin_end, vector<int> seamPath, bool shift2end) ;
    // 获取displacementField
    // vector< vector<CoordinateDouble>> get_displacementField(Mat& src, Mat& mask);
    vector< vector<CoordinateInt>> get_displacementField(Mat& src, Mat& mask);
    // 获取标准化矩阵网格
    vector< vector<CoordinateDouble>> get_rectangleMesh(Mat& src, Config config);
    // 获取变形后的网格
    void wrap_Back(vector< vector<CoordinateDouble>>& mesh,  const vector< vector<CoordinateInt>>& displacementField, const Config& config);

    void draw_Mesh(Mat& src,  vector< vector<CoordinateDouble>>& mesh, Config config);
};


#endif //LOCALWRAPPPING_h