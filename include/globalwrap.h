#pragma once

# ifndef GLOBALWRAP_H
# define GLOBALWRAP_H

#include "config.h"
#include <vector>
#include <Eigen/Sparse> // 稀疏矩阵
#include <Eigen/Dense> // 稠密矩阵
#include <iostream>

using std::vector;
using std::pair;
#define clampValue(x, a, b) a < b? (x < a? a : (x > b? b : x)) : (x < b? b : (x < a? a : a))

typedef Eigen::SparseMatrix<double> SpMat; // 稀疏矩阵
typedef Eigen::VectorXd VectorXd;  //表示一个动态大小的double精度浮点数向量。
typedef Eigen::MatrixXd MatrixXd;  // 表示一个动态大小的double精度浮点数矩阵。
typedef Eigen::Vector2d Vector2d; // 表示一个2维的double精度浮点数向量。
typedef Eigen::Matrix2d Matrix2d;  // 表示一个2维的double精度浮点数矩阵。


class BilinearWeights{
public:
    double s, t;
    BilinearWeights(double S, double T);
};

class  GlobalWrap{
private:
    /**
     * 应用lsd算法检测源图像中的直线
     */
    vector< StraightLine> detect_Src_StraightLines(const cv::Mat& src, const cv::Mat& mask);

    /**
     * 判断点point是否在quad内部(包括与边界重合情况)
     */
    bool is_PointInside_Quad(const CoordinateDouble& point, const vector<CoordinateDouble>& quadVertexes);

    /**
     * 判断两线段是否相交，若相交，计算出交点
     */
    bool is_LineSegments_Intersect(const StraightLine& line1, const StraightLine& line2, CoordinateDouble& intersection);

    /**
     * 计算 直线 与 quad 的交点
     */
    vector<CoordinateDouble> get_IntersectionWithQuad(const StraightLine& line, const vector<CoordinateDouble>& quadVertexes);

    /**
     * 计算 点point与quad的(逆)双线性插值系数
     */
    BilinearWeights get_BiWeights_PointQuad(const CoordinateDouble& point, const vector<CoordinateDouble>& quadVertexes);

    /**
     * 根据BilinearWeights中的s和t值,计算出4个权重系数v1w、v2w、v3w、v4w,并将它们填充到一个2x8的矩阵。
     */
    MatrixXd BilinearWeights2Matrix(BilinearWeights& weight);

    /**
     * 在原矩阵上按对角进行扩展,即矩阵的合并,返回扩展之后的矩阵
     */
    SpMat SpMat_extendByDiagonal(const SpMat& mat, const MatrixXd& add, int QuadIdx, const Config& config);


public:
    
    /**
     * shape energy
     * 获取shape energy的系数矩阵
    */ 
    SpMat get_ShapeE_Coeff(const vector< vector<CoordinateDouble>>& mesh, const Config& config);
    

    /**
     * 优化 ES(q) = (1/N) * Σq || (Aq * ((Aq^T * Aq)^-1) * Aq^T - I) * Vq ||^2 计算过程
     * 此时我们构造一个向量V，包含所有顶点的坐标x和y，大小为2N * 1
     * 每个quad的Vq向量就可以通过一个选择矩阵Qq获得，Qq的大小为8 * 2N，使得Vq = Qq * V
     * ES(v) = (1/N) * Σq || (Aq * ((Aq^T * Aq)^-1) * Aq^T - I) * Qq^T * V ||^2
     *       = (1/N) * || (Q^T * (Aq(Aq^T Aq)^(-1) Aq^T - I) * Q) * V ||^2
     * 这样可以直接优化网格的顶点坐标向量V
     */
    SpMat get_SelectMatrix_Q(const vector< vector<CoordinateDouble>>& mesh, const Config& config);

    /**
     * preprocess line 
     * 裁剪直线段，让直线段位于quad内部
     */
    vector< vector< vector<StraightLine>>> cut_LineSegmentsWithQuad(const vector< vector<CoordinateDouble>>& mesh, vector<StraightLine>& lines, const Config& config);
    
    /**
     * preprocess line 
     * 初始化线段分割,将倾斜角度位于同一个bin的线段分配到一个集合中
     */
    vector< vector< vector<StraightLine>>> init_LineSegments(const vector< vector<CoordinateDouble>>& mesh, cv::Mat& mask, cv::Mat&src, const Config& config,
        vector< double> &lineIdx_Theta, vector<int>& lineIdx_BinIdx, vector<double>& rotate_theta);


    /**
     * line energy
     * C = R * ehat * (ehat^T * ehat)^-1 * ehat^T * R^T - I
     * e = line_startP_Weight_mat - line_startP_Weight_mat
     * [Matrix1
     *      Matrix2
     *          ······
     *              MatrixN] N = quadNums;
     * Matrix1 = [C_e1  ······  C_en]^T 表示第一个quad内部所有直线的C*e
     * 获取直线保持函数的相关矩阵
     */
    SpMat get_LineE_Matrix(const vector< vector<CoordinateDouble>>& mesh, cv::Mat& mask, vector<double>& rotate_theta, 
        vector< vector< vector<StraightLine>>>& lineSegmentsInQuad, vector<pair<MatrixXd, MatrixXd>>& BiWeightsVec,
        vector<bool>& bad, int& lineNum, const Config& config );
    
    /**
     * Boundary energy
     * 此时我们根据坐标向量V构造一个选择矩阵Q_boundary，大小为2N * 2N，使得Q_boundary * V即为边界坐标
     * 同时构造一个理想边界Boundary，使得Eb(V) = || Boundary - (Q_boundary * V) ||^2
     */
    void get_BoundaryE_Matrix(const vector< vector<CoordinateDouble>>& mesh, const Config& config, VectorXd& Boundary, SpMat& Q_boundary);


    /**
     * 获取(row, col)处的坐标向量Vq
     */
    VectorXd get_Vq(const vector< vector<CoordinateDouble>>& mesh, int row, int col);


    /**
     * 可视化Straight line
     */
    void Show_StraightLines(cv::Mat src, vector< vector< vector<StraightLine>>>& lineSegmentsInQuad, const Config& config);
};




# endif // GLOBALWRAP_H