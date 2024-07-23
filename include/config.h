#pragma once  //确保头文件只被编译一次

#ifndef CONFIG_H
#define CONFIG_H


#include <vector>
#include "GL/glut.h"
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

#include <Eigen/SparseCholesky>
#include <Eigen/Sparse> // 稀疏矩阵
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

using std::vector;

#define DEBUG 0
#define MAX_COST 1e8
#define PI 3.14159265358979323846
// #define INF 1e8


typedef Eigen::SparseMatrix<double> SpMat; // 稀疏矩阵(默认行主序)


// 配置类
class Config{
public:
    int thetaBins = 50;
    int iters = 10; // 迭代次数 
    double LambdaS = 10, LambdaL = 1000, LambdaB = 1e8;
    int meshRows = 20, meshCols = 20; // 网格行数，列数


    int rows, cols; // 图片行数， 列数
    int quadRows, quadCols; // 网格中四边形行数，列数
    double meshRowSize, meshColSize; // 网格跨度(跨过的像素数目)

    // 构造函数
    Config(){}
    Config(int rows, int cols){
        this->rows = rows;  this->cols = cols;
        this->quadRows = meshRows - 1; this->quadCols = meshCols - 1;
        this->meshRowSize = double(rows - 1) / double(quadRows);  // 像素数目 / 四边形数目
        this->meshColSize = double(cols - 1) / double(quadCols);  // 像素数目 / 四边形数目
    }
};


// 坐标类 coordinate
template <typename T>
class Coordinate{
public:
    T row; T col;

    // 构造函数
    Coordinate() : row(0), col(0) {}
    Coordinate(T Row, T Col) : row(Row), col(Col) {}

    // 重载 < 运算符
    bool operator<(const Coordinate& c){
        return row < c.row? true : (row > c.row? false : col < c.col);
    }

    // 重载 > 运算符
    bool operator>(const Coordinate& c){
        return row > c.row? true : (row < c.row? false : col > c.col);
    }

    // 重载 == 运算符
    bool operator==(const Coordinate& c){
        return row == c.row && col == c.col; 
    }
    // 重载 - 运算符
    Coordinate operator-(const Coordinate& other) const {
        return Coordinate(row - other.row, col - other.col);
    }

    Coordinate operator+(const Coordinate& other) const {
        return Coordinate(row + other.row, col + other.col);
    }
    // 重载输出
    friend std::ostream& operator<<(std::ostream& stream, const Coordinate& c){
        stream << "(" << c.row << ", " << c.col << ")" <<std::endl;
        return stream;
    }
};

using CoordinateInt = Coordinate<int>;  // 像素坐标
using CoordinateDouble = Coordinate<double>;   // 四边形坐标


//直线
class StraightLine{
public:
	CoordinateDouble p1, p2;

	//构造函数
	StraightLine(double row1, double col1, double row2, double col2) : p1(row1, col1), p2(row2, col2){}
	StraightLine() : p1(0, 0), p2(0, 0){}
	StraightLine(CoordinateDouble P1, CoordinateDouble P2) : p1(P1), p2(P2){}
};

cv::Mat Init_Mask(const cv::Mat& src);

void draw_StraightLine(cv::Mat& src, const StraightLine& line);

/**
 * 将两个稀疏系数矩阵按行合并
 * @param mat1 第一个稀疏矩阵
 * @param mat2 第二个稀疏矩阵
 * @return 合并后的稀疏矩阵
 */
SpMat mergeMatricesByRow(const SpMat& mat1, const SpMat& mat2);
void scale_mesh(vector<vector<CoordinateDouble>>& mesh, double enlargeFacrtor_row, double  enlargeFacrtor_col, Config config);

// GLuint matToTexture(cv::Mat& mat, GLenum minFilter = GL_LINEAR, GLenum magFilter = GL_LINEAR, GLenum wrapFilter = GL_CLAMP);


#endif // CONFIG_H