#include "config.h"


using namespace cv;
using namespace std;


Mat Init_Mask(const Mat& img){

    Mat mask;
    Vec3b Corner = img.at<Vec3b>(0);
    const double cornerEps = 300;
    mask.create(img.size(), CV_8UC1);
    mask.setTo(Scalar(255));

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
        for (int j = img.cols - 1; j >= 0; j--) {
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
    }

    for (int j = 0; j < img.cols; j++) {
        for (int i = 0; i < img.rows; i++) {
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
        for (int i = img.rows - 1; i >= 0; i--) {
            Vec3d de = img.at<Vec3b>(i, j);
            de -= (Vec3d)Corner;
            if (de.dot(de) < cornerEps) {
                mask.at<uchar>(i, j) = 0;
            }
            else break;
        }
    }

    Mat erodeImg, dilateImg;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    erode(mask, erodeImg, element);
    erode(erodeImg, erodeImg, element);

    mask = erodeImg;

    // cv::imshow("mask", mask);
    // cv::waitKey(0);

    // 翻转 mask 像素
    bitwise_not(mask, mask);

    // cv::imshow("mask", mask);
    // cv::waitKey(0);
    
    return mask;
}


void draw_StraightLine(Mat& src, const StraightLine& line)
{
    Point p1(line.p1.col, line.p1.row);
    Point p2(line.p2.col, line.p2.row);
	cv::line(src, p1, p2, cv::Scalar(0, 255, 0), 1, 1);
}

/**
 * 将两个稀疏系数矩阵按行合并
 * @param mat1 第一个稀疏矩阵
 * @param mat2 第二个稀疏矩阵
 * @return 合并后的稀疏矩阵
 */
SpMat mergeMatricesByRow(const SpMat& mat1, const SpMat& mat2) {
    // 计算合并后矩阵的行数和列数
    int totalRows = mat1.rows() + mat2.rows();
    int totalCols = std::max(mat1.cols(), mat2.cols());

    // 创建合并后的稀疏矩阵
    SpMat combinedMatrix(totalRows, totalCols);

    // 将第一个矩阵的元素复制到合并矩阵
    int currentRow = 0;
    for (int k = 0; k < mat1.outerSize(); ++k) {
        for (SpMat::InnerIterator it(mat1, k); it; ++it) {
            combinedMatrix.insert(currentRow + it.row(), it.col()) = it.value();
        }
    }
    currentRow += mat1.rows();

    // 将第二个矩阵的元素复制到合并矩阵
    for (int k = 0; k < mat2.outerSize(); ++k) {
        for (SpMat::InnerIterator it(mat2, k); it; ++it) {
            combinedMatrix.insert(currentRow + it.row(), it.col()) = it.value();
        }
    }

    // 压缩合并后的稀疏矩阵
    combinedMatrix.makeCompressed();
    return combinedMatrix;
}

/*
	扩充网格,
	根据扩充因子scaleFacrtor来相应将坐标扩充scaleFacrtor倍
*/

using std::vector;
void scale_mesh(vector<vector<CoordinateDouble>>& mesh, double enlargeFacrtor_row, double  enlargeFacrtor_col, Config config)
{
	int numMeshRow = config.meshRows;
	int numMeshCol = config.meshCols;
	for (int row = 0; row < numMeshRow; row++)
	{
		for (int col = 0; col < numMeshCol; col++)
		{
			CoordinateDouble& coord = mesh[row][col];
			coord.row = coord.row * enlargeFacrtor_row;//row * enlargeFacrtor
			coord.col = coord.col * enlargeFacrtor_col;//col * enlargeFacrtor
		}
	}
}