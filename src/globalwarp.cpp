#include "globalwarp.h"
#include "lsd.h"
#include "config.h"


using namespace cv;
using namespace std;

BilinearWeights::BilinearWeights(double S, double T): s(S), t(T) {}

/**
 * 应用lsd算法检测源图像中的直线
 */
vector< StraightLine> Globalwarp::detect_Src_StraightLines(const Mat& src, const Mat& mask){
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    
    int rows = src.rows, cols = src.cols;
    double* image = new double[src_gray.rows * src_gray.cols];
    for(int i=0; i<src_gray.rows; i++){
        for(int j=0; j<src_gray.cols; j++){
            image[i * src_gray.cols + j] = src_gray.at<uchar>(i, j);
        }
    }

    vector< StraightLine> lines;
    int lineNum; // 检测到的直线数量
    double* out = lsd(&lineNum, image, src_gray.cols, src_gray.rows);

    Mat src_copy;
    src.copyTo(src_copy);
    if(DEBUG){
        // 交互
        namedWindow("Src_lines", WINDOW_NORMAL);
        
    }

    for(int i=0; i<lineNum; i++){
        StraightLine line(out[i*7+1], out[i*7+0], out[i*7+3], out[i*7+2]);

        // 验证两个端点在图像内
        if(mask.at<uchar>(line.p1.row, line.p1.col) == 0 && mask.at<uchar>(line.p2.row, line.p2.col) == 0){
            lines.push_back(line);
            if(DEBUG) draw_StraightLine(src_copy, line);
        }
        if(DEBUG){
            imshow("Src_lines", src_copy);
            waitKey(1);
        }
    }
    if(DEBUG){
        waitKey(0);
        destroyAllWindows();
    }
    return lines;
}

/**
 * 判断点point是否在quad内部
 */
bool Globalwarp::is_PointInside_Quad(const CoordinateDouble& point, const vector<CoordinateDouble>& quadVertexes){
    Vector2d topLeft(quadVertexes[0].col, quadVertexes[0].row);
    Vector2d topRight(quadVertexes[1].col, quadVertexes[1].row);
    Vector2d bottomLeft(quadVertexes[2].col, quadVertexes[2].row);
    Vector2d bottomRight(quadVertexes[3].col, quadVertexes[3].row);
    Vector2d p(point.col, point.row);

    Vector2d v1 = topRight - topLeft;
    Vector2d v2 = bottomRight - topRight;
    Vector2d v3 = bottomLeft - bottomRight;
    Vector2d v4 = topLeft - bottomLeft;

    Vector2d p1 = p - topLeft;
    Vector2d p2 = p - topRight;
    Vector2d p3 = p - bottomRight;
    Vector2d p4 = p - bottomLeft;

    double c1 = v1.x() * p1.y() - v1.y() * p1.x();
    double c2 = v2.x() * p2.y() - v2.y() * p2.x();
    double c3 = v3.x() * p3.y() - v3.y() * p3.x();
    double c4 = v4.x() * p4.y() - v4.y() * p4.x();

    bool has_neg = (c1 < 0) || (c2 < 0) || (c3 < 0) || (c4 < 0);
    bool has_pos = (c1 > 0) || (c2 > 0) || (c3 > 0) || (c4 > 0);
    bool on_boundary = (c1 == 0) || (c2 == 0) || (c3 == 0) || (c4 == 0);

    return !(has_neg && has_pos) || on_boundary;
}

bool Globalwarp::is_LineSegments_Intersect(const StraightLine& line1, const StraightLine& line2, CoordinateDouble& intersection) {
    double p1_x = line1.p1.col, p1_y = line1.p1.row, p2_x = line1.p2.col, p2_y = line1.p2.row;
    double p3_x = line2.p1.col, p3_y = line2.p1.row, p4_x = line2.p2.col, p4_y = line2.p2.row;

    // 当一条直线平行于 y 轴时
    if (fabs(p1_x - p2_x) < 1e-6) {
        // 判断另一条直线是否也平行于 y 轴
        if (fabs(p3_x - p4_x) < 1e-6) {
            // 如果两条直线都平行于 y 轴,则不相交
            intersection = CoordinateDouble(0, 0);
            return false;
        } else {
            // 计算交点的 x 坐标
            double x = p1_x;
            double y = (p4_y - p3_y) / (p4_x - p3_x) * (x - p3_x) + p3_y;
            
            // 检查交点是否在两条线段的范围内
            bool x_in_range = (y >= std::min(p1_y, p2_y) && y <= std::max(p1_y, p2_y)) &&
                              (y >= std::min(p3_y, p4_y) && y <= std::max(p3_y, p4_y));
            if (x_in_range) {
                intersection = CoordinateDouble(y, x);
                return true;
            } else {
                intersection = CoordinateDouble(0, 0);
                return false;
            }
        }
    }
    
    // 当一条直线平行于 x 轴时
    if (fabs(p1_y - p2_y) < 1e-6) {
        // 判断另一条直线是否也平行于 x 轴
        if (fabs(p3_y - p4_y) < 1e-6) {
            // 如果两条直线都平行于 x 轴,则不相交
            intersection = CoordinateDouble(0, 0);
            return false;
        } else {
            // 计算交点的 y 坐标
            double y = p1_y;
            double x = (p4_x - p3_x) / (p4_y - p3_y) * (y - p3_y) + p3_x;
            
            // 检查交点是否在两条线段的范围内
            bool y_in_range = (x >= std::min(p1_x, p2_x) && x <= std::max(p1_x, p2_x)) &&
                              (x >= std::min(p3_x, p4_x) && x <= std::max(p3_x, p4_x));
            if (y_in_range) {
                intersection = CoordinateDouble(y, x);
                return true;
            } else {
                intersection = CoordinateDouble(0, 0);
                return false;
            }
        }
    }

    // 计算直线的斜率
    double k1 = (p2_y - p1_y) / (p2_x - p1_x);
    double k2 = (p4_y - p3_y) / (p4_x - p3_x);

    // 判断是否平行
    if (fabs(k1 - k2) < 1e-6) {
        intersection = CoordinateDouble(0, 0);
        return false;
    }

    double x = (p3_y - p1_y + k1 * p1_x - k2 * p3_x) / (k1 - k2);
    double y = k1 * (x - p1_x) + p1_y;

    bool x_in_range = (x >= std::min(p1_x, p2_x) && x <= std::max(p1_x, p2_x)) &&
                      (x >= std::min(p3_x, p4_x) && x <= std::max(p3_x, p4_x));
    bool y_in_range = (y >= std::min(p1_y, p2_y) && y <= std::max(p1_y, p2_y)) &&
                      (y >= std::min(p3_y, p4_y) && y <= std::max(p3_y, p4_y));

    // 检查交点是否在两条线段的范围内
    if (x_in_range && y_in_range) {
        intersection = CoordinateDouble(y, x);
        return true;
    } else {
        intersection = CoordinateDouble(0, 0);
        return false;
    }
}

/**
 * 计算 直线 与 quad 的交点
 */
vector< CoordinateDouble> Globalwarp::get_IntersectionWithQuad(const StraightLine& line, const vector<CoordinateDouble>& quadVertexes){
    vector< CoordinateDouble> intersections;
    CoordinateDouble topLeftPoint = quadVertexes[0], topRightPoint = quadVertexes[1], bottomLeftPoint = quadVertexes[2], bottomRightPoint = quadVertexes[3];
    vector< StraightLine> quadBoundaries = { StraightLine(topLeftPoint, topRightPoint), 
        StraightLine(topRightPoint, bottomRightPoint),
        StraightLine(bottomRightPoint, bottomLeftPoint),
        StraightLine(bottomLeftPoint, topLeftPoint) };
    for(auto& boundary : quadBoundaries){
        CoordinateDouble intersection;
        if(is_LineSegments_Intersect(line, boundary, intersection)){
            intersections.push_back(intersection);
        }
    }
    return intersections;
}

double cross(CoordinateDouble a, CoordinateDouble b){ return a.col*b.row - a.row*b.col; }

/**
 * 计算 点point与quad的双线性插值系数
 */
BilinearWeights Globalwarp::get_BiWeights_PointQuad(const CoordinateDouble& point, const vector<CoordinateDouble>& quadVertexes){
    CoordinateDouble p1 = quadVertexes[0], p2 = quadVertexes[1], p3 = quadVertexes[2], p4 = quadVertexes[3];
    double eps = 1e-6;

    // version 1 
    // 计算四边形边界的斜率
    // double slopeTop    = (p2.row - p1.row) / (p2.col - p1.col + eps);//上边界斜率
    // double slopeBottom = (p4.row - p3.row) / (p4.col - p3.col + eps);//下边界斜率
    // double slopeLeft   = (p1.row - p3.row) / (p1.col - p3.col + eps);//左边界斜率
    // double slopeRight  = (p2.row - p4.row) / (p2.col - p4.col + eps);//右边界斜率

    // if(fabs(slopeBottom - slopeTop) < eps && fabs(slopeLeft - slopeRight) < eps){ // 平行四边形，采用方法3
    //     Matrix2d mat1; //[x2-x1, x3-x1; y2-y1, y3-y1]
    //     mat1 << p2.col - p1.col, p3.col - p1.col,
    //             p2.row - p1.row, p3.row - p1.row;

    //     if(mat1.determinant() != 0) {
    //         MatrixXd mat2(2, 1);
    //         mat2 << point.col - p1.col, point.row - p1.row;

    //         MatrixXd solutionMatrix = mat1.inverse() * mat2;

    //         return BilinearWeights(solutionMatrix(0, 0), solutionMatrix(1, 0));
    //     } else {
    //         return BilinearWeights(0, 0);
    //     }
    // } else if(fabs(slopeLeft - slopeRight) < eps) { // 左右边界平行，采用方法2
    //     double a = (p2.col - p1.col) * (p4.row - p3.row) - (p2.row - p1.row) * (p4.col - p3.col);
    //     double b = point.row * ((p4.col - p3.col) - (p2.col - p1.col)) - point.col * ((p4.row - p3.row) - (p2.row - p1.row)) 
    //                + p1.col * (p4.row - p3.row) - p1.row * (p4.col - p3.col) + p3.row * (p2.col - p1.col) - p3.col * (p2.row - p1.row);
    //     double c = point.row * (p3.col - p1.col) - point.col * (p3.row - p1.row) + p1.col * p3.row - p3.col * p1.row;

    //     double discriminant = b * b - 4 * a * c;
    //     if (discriminant >= 0) {
    //         double sqrtDiscriminant = sqrt(discriminant);
    //         double s1 = (-b + sqrtDiscriminant) / (2 * a);
    //         double s2 = (-b - sqrtDiscriminant) / (2 * a);
    //         double S = (s1 >= 0 && s1 <= 1) ? s1 : (s2 >= 0 && s2 <= 1 ? s2 : 0);
    //         S = clampValue(S, 0, 1);

    //         double val = (p3.row + (p4.row - p3.row) * S - p1.row - (p2.row - p1.row) * S);
    //         double T = (point.row - p1.row - (p2.row - p1.row) * S) / val;
    //         if(fabs(val) < eps){
    //             T = (point.col - p1.col - (p2.col - p1.col) * S) / (p3.col + (p4.col - p3.col) * S - p1.col - (p2.col - p1.col) * S);
    //         }

    //         return BilinearWeights(S, T);
    //     } else {
    //         return BilinearWeights(0, 0);
    //     }
    // } else { // 上下边界平行或者都不平行 方法1
    //     double a = (p3.col - p1.col) * (p4.row - p2.row) - (p3.row - p1.row) * (p4.col - p2.col);
    //     double b = point.row * ((p4.col - p2.col) - (p3.col - p1.col)) - point.col * ((p4.row - p2.row) - (p3.row - p1.row)) 
    //                + (p3.col - p1.col) * (p2.row) - (p3.row - p1.row) * (p2.col) + (p1.col) * (p4.row - p2.row) - (p1.row) * (p4.col - p2.col);
    //     double c = point.row * (p2.col - p1.col) - (point.col) * (p2.row - p1.row) + p1.col * p2.row - p2.col * p1.row;

    //     double discriminant = b * b - 4 * a * c;
    //     if (discriminant >= 0) {
    //         double sqrtDiscriminant = sqrt(discriminant);
    //         double t1 = (-b + sqrtDiscriminant) / (2 * a);
    //         double t2 = (-b - sqrtDiscriminant) / (2 * a);
    //         double T = (t1 >= 0 && t1 <= 1) ? t1 : (t2 >= 0 && t2 <= 1 ? t2 : 0);
    //         T = clampValue(T, 0, 1);

    //         double val = (p2.row + (p4.row - p2.row) * T - p1.row - (p3.row - p1.row) * T);
    //         double S = (point.row - p1.row - (p3.row - p1.row) * T) / val;
    //         if (fabs(val) < eps) {
    //             S = (point.col - p1.col - (p3.col - p1.col) * T) / (p2.col + (p4.col - p2.col) * T - p1.col - (p3.col - p1.col) * T);
    //         }

    //         return BilinearWeights(clampValue(S, 0, 1), clampValue(T, 0, 1));
    //     } else {
    //         return BilinearWeights(0, 0);
    //     }
    // }

    // version 2 逆线性插值
    CoordinateDouble e = p2 - p1;
    CoordinateDouble f = p3 - p1;
    CoordinateDouble g = p1 - p2 + p4 - p3;
    CoordinateDouble h = point - p1;


    double k2 = cross(g, f);
	double k1 = cross(e, f) + cross(h, g);
	double k0 = cross(h, e);

    	double u, v;

	if ((int)k2 == 0)
	{
		v = -k0 / k1;
		u = (h.col - f.col*v) / (e.col + g.col*v);
	}
	else
	{
		double w = k1 * k1 - 4.0*k0*k2;
        if(w <= 0.0){
            return BilinearWeights(-1.0, -1.0);
        } 
		assert(w >= 0.0);
        
		w = sqrt(w);

		double v1 = (-k1 - w) / (2.0*k2);
		double u1 = (h.col - f.col*v1) / (e.col + g.col*v1);

		double v2 = (-k1 + w) / (2.0*k2);
		double u2 = (h.col - f.col*v2) / (e.col + g.col*v2);

		u = u1;
		v = v1;

		if (v<0.0 || v>1.0 || u<0.0 || u>1.0) { u = u2;   v = v2; }
		if (v-0.0 || v>1.0 || u<0.0 || u>1.0) 
		{

			u = -1.0; 
			v = -1.0; 
		}
	}
	return BilinearWeights(u, v);
}

/**
 * 根据BilinearWeights中的s和t值,计算出4个权重系数v1w、v2w、v3w、v4w,并将它们填充到一个2x8的矩阵。
 */
MatrixXd Globalwarp::BilinearWeights2Matrix(BilinearWeights& w){
    MatrixXd mat(2, 8);
    double v1w = 1 - w.s - w.t + w.s * w.t;//1-s-t-st
	double v2w = w.s - w.s * w.t;//s-st
	double v3w = w.t - w.s * w.t;//t-st
	double v4w = w.s * w.t;//st
	mat << v1w, 0, v2w, 0, v3w, 0, v4w, 0,
		   0, v1w, 0, v2w, 0, v3w, 0, v4w;
	return mat;
}

/**
 * 在原矩阵上按对角进行扩展,即矩阵的合并,返回扩展之后的矩阵
 */
SpMat Globalwarp::SpMat_extendByDiagonal(const SpMat& mat, const MatrixXd& add, int QuadIdx, const Config& config) {
    int total_cols = 8 * config.quadRows * config.quadCols;
    SpMat res(mat.rows() + add.rows(), total_cols);

    // 预留空间以提高性能
    res.reserve(mat.nonZeros() + add.rows() * add.cols());

    // 复制原始稀疏矩阵
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (SpMat::InnerIterator it(mat, k); it; ++it) {
            res.insert(it.row(), it.col()) = it.value();
        }
    }

    int lefttop_row = mat.rows(); // add开始的行数
    int lefttop_col = 8 * QuadIdx; // add开始的列数

    // 插入新的稠密矩阵
    for (int row = 0; row < add.rows(); ++row) {
        for (int col = 0; col < add.cols(); ++col) {
            if (add(row, col) != 0) { // 只插入非零元素
                res.insert(lefttop_row + row, lefttop_col + col) = add(row, col);
            }
        }
    }

    res.makeCompressed(); // 对稀疏矩阵进行压缩
    return res;
}


/**
 * 获取shape energy的系数矩阵
 */
SpMat Globalwarp::get_ShapeE_Coeff(const vector< vector<CoordinateDouble>>& mesh, const Config& config){
    int meshRows = config.meshRows, meshCols = config.meshCols;
    int quadRows = config.quadRows, quadCols = config.quadCols;

    // coeff = Aq * ((Aq^T * Aq)^-1) * Aq^T - I;
    /*
     *  Coeff = [corff0
     *              coeff1
     *                  coeff2
     *                      ·····
     *                          coeffn]
     */
    // 由于每个四边形单元格对应一个8x8的系数矩阵coeff，而网格中总共有quadRows * quadCols个四边形单元格。
    // 所以矩阵的行、列数需要是8 * quadRows * quadCols。
    SpMat ShapeE_Coeff(8 * quadRows * quadCols, 8 * quadRows * quadCols);

    for(int i=0; i<meshRows-1; i++){
        for(int j=0; j<meshCols-1; j++){
            CoordinateDouble p0 = mesh[i][j], p1 = mesh[i][j+1], p3 = mesh[i+1][j+1], p2 = mesh[i+1][j];
            
            MatrixXd Aq(8, 4);  // Aq相似变换矩阵
            Aq << p0.col, -p0.row, 1, 0,
                  p0.row, p0.col, 0, 1,
                  p1.col, -p1.row, 1, 0,
                  p1.row, p1.col, 0, 1,
                  p2.col, -p2.row, 1, 0,
                  p2.row, p2.col, 0, 1,
                  p3.col, -p3.row, 1, 0,
                  p3.row, p3.col, 0, 1;
            
            MatrixXd AqT = Aq.transpose(); // Aq^T
            MatrixXd AqT_Aq_inverse = (AqT * Aq).inverse(); // (Aq^T * Aq)^-1
            MatrixXd coeff = Aq * AqT_Aq_inverse * AqT - MatrixXd::Identity(8, 8); // Aq * ((Aq^T * Aq)^-1) * Aq^T - I;

            int startIdx_row = i * 8 * quadCols + j * 8;  // 找到该稀疏矩阵的开始坐标
            for(int k=0; k<8; k++){
                for(int l=0; l<8; l++){
                    ShapeE_Coeff.insert(startIdx_row + k, startIdx_row + l) = coeff(k, l);
                }
            }
        }
    }
    ShapeE_Coeff.makeCompressed(); // 压缩稀疏矩阵
    return ShapeE_Coeff;
}

/**
     * preprocess line 
     * 裁剪直线段，让直线段位于quad内部
     */
    vector< vector< vector<StraightLine>>> Globalwarp::cut_LineSegmentsWithQuad(const vector< vector<CoordinateDouble>>& mesh, 
        vector<StraightLine>& lines, const Config& config){
            int quadRows = config.quadRows, quadCols = config.quadCols;

            vector< vector< vector<StraightLine>>> lineSegments_InQuad(quadRows, vector< vector<StraightLine>>(quadCols));
            for(int i = 0; i < quadRows; i++){
                for(int j=0; j < quadCols; j++){
                    vector< StraightLine> lines_curQuad;
                    vector< CoordinateDouble> quadVertexes = { mesh[i][j], mesh[i][j+1], mesh[i+1][j], mesh[i+1][j+1] };

                    for(auto& line : lines){
                        if(is_PointInside_Quad(line.p1, quadVertexes) && is_PointInside_Quad(line.p2, quadVertexes)){
                            lines_curQuad.push_back(line);
                        }else if(is_PointInside_Quad(line.p1, quadVertexes) && !is_PointInside_Quad(line.p2, quadVertexes)){
                            vector<CoordinateDouble> intersections = get_IntersectionWithQuad(line, quadVertexes);
                            for(auto& intersection : intersections){
                                // printf("intersection: (%f, %f)\n", intersection.col, intersection.row);
                                if(intersection.col != line.p1.col || intersection.row != line.p1.row){
                                    lines_curQuad.push_back(StraightLine(line.p1, intersection));
                                }
                            }
                            // if(intersections.size() == 1) lines_curQuad.push_back(StraightLine(line.p1, intersections[0]));
                        }else if(!is_PointInside_Quad(line.p1, quadVertexes) && is_PointInside_Quad(line.p2, quadVertexes)){
                            vector<CoordinateDouble> intersections = get_IntersectionWithQuad(line, quadVertexes);

                            for(auto& intersection : intersections){
                                // printf("intersection: (%f, %f)\n", intersection.col, intersection.row);
                                if(intersection.col != line.p2.col || intersection.row != line.p2.row){
                                    lines_curQuad.push_back(StraightLine(intersections[0], line.p2));
                                }
                            }
                            // if(intersections.size() == 1) lines_curQuad.push_back(StraightLine(intersections[0], line.p2));
                        }else{
                            vector<CoordinateDouble> intersections = get_IntersectionWithQuad(line, quadVertexes);
                            if(intersections.size() == 2){
                                // printf("intersection: (%f, %f)\n", intersections[0].col, intersections[0].row);
                                // printf("intersection: (%f, %f)\n", intersections[1].col, intersections[1].row);
                                lines_curQuad.push_back(StraightLine(intersections[0], intersections[1]));
                            }
                        }
                    }
                    lineSegments_InQuad[i][j] = lines_curQuad;
                }
            }
            return lineSegments_InQuad;
        }

/**
 * preprocess line 
 * 初始化线段分割,将倾斜角度位于同一个bin的线段分配到一个集合中
 */
vector< vector< vector<StraightLine>>> Globalwarp::init_LineSegments(const vector< vector<CoordinateDouble>>& mesh, cv::Mat& mask, cv::Mat&src, const Config& config,
        vector< double> &lineIdx_Theta, vector<int>& lineIdx_BinIdx, vector<double>& rotate_theta){
            double thetaPerbin = PI / config.thetaBins;

            vector<StraightLine> lines = detect_Src_StraightLines(src, mask);
            vector< vector< vector<StraightLine>>> lineSegmentsInQuad = cut_LineSegmentsWithQuad(mesh, lines, config);
            
            int lineIdx = 0;
            for(int i=0; i<config.quadRows; i++){
                for(int j=0; j<config.quadCols; j++){
                    for(auto& line : lineSegmentsInQuad[i][j]){
                        // atan值域为: [-pi/2, pi/2]
                        double theta = atan((line.p1.row - line.p2.row) / (line.p1.col - line.p2.col));
                        int BinIdx = ((theta + (PI / 2)) / thetaPerbin);
                        lineIdx_Theta.push_back(theta);
                        lineIdx_BinIdx.push_back(BinIdx);
                        rotate_theta.push_back(0);
                    }
                }
            }
            return lineSegmentsInQuad;
        }

/**
 * 获取(row, col)处的坐标向量
 */
VectorXd Globalwarp::get_Vq(const vector< vector<CoordinateDouble>>& mesh, int i, int j){
    VectorXd Vq(8);
    CoordinateDouble p0 = mesh[i][j], p1 = mesh[i][j+1], p3 = mesh[i+1][j+1], p2 = mesh[i+1][j];
    Vq << p0.col, p0.row, p1.col, p1.row, p2.col, p2.row, p3.col, p3.row;
    return Vq;
}

/**
 * line energy
 * 获取直线保持函数的系数矩阵 C
 */
SpMat Globalwarp::get_LineE_Matrix(const vector< vector<CoordinateDouble>>& mesh, Mat& mask, vector<double>& rotate_theta, 
        vector< vector< vector<StraightLine>>>& lineSegmentsInQuad, vector<pair<MatrixXd, MatrixXd>>& BiWeightsVec,
        vector<bool>& bad, int& lineNum, const Config& config ){
            lineNum = -1; 
            int rows = config.rows, cols = config.cols;
            int meshRows = config.meshRows, meshCols = config.meshCols;
            int quadRows = config.quadRows, quadCols = config.quadCols;
            SpMat LineE;
            for(int i=0; i<quadRows; i++){
                for(int j=0; j<quadCols; j++){
                    vector<StraightLine> lineSegments_CurQuad = lineSegmentsInQuad[i][j];
                    int quadIdx = i * quadCols + j;
                    if(!lineSegments_CurQuad.size()) continue;

                    vector<CoordinateDouble> quadVertexes = {mesh[i][j], mesh[i][j+1], mesh[i+1][j], mesh[i+1][j+1]};

                    MatrixXd C_quad_rowStack(0, 8); // C = R * ehat * (ehat^T * ehat)^-1 * ehat^T * R^T - I
                    for(int k = 0; k<lineSegments_CurQuad.size(); k++){
                        lineNum ++;
                        StraightLine line = lineSegments_CurQuad[k];
                        CoordinateDouble startPoint = line.p1,  endPoint = line.p2;
                        BilinearWeights startPoint_Weights = get_BiWeights_PointQuad(startPoint, quadVertexes);
                        MatrixXd startPoint_Weights_Mat = BilinearWeights2Matrix(startPoint_Weights);
                        
                        BilinearWeights endPoint_weights = get_BiWeights_PointQuad(endPoint, quadVertexes);
                        MatrixXd endPoint_weights_Mat = BilinearWeights2Matrix(endPoint_weights);

                        // 测试系数矩阵
                        VectorXd Vq = get_Vq(mesh, i, j);
                        Vector2d startp_ans = startPoint_Weights_Mat * Vq - Vector2d(startPoint.col, startPoint.row);
                        Vector2d endp_ans = endPoint_weights_Mat * Vq - Vector2d(endPoint.col, endPoint.row);
                        if((startp_ans.norm() > 0.00001 || endp_ans.norm() > 0.00001)){
                            //错误情况
                            bad.push_back(true);
                            BiWeightsVec.push_back(make_pair(MatrixXd::Zero(2, 8), MatrixXd::Zero(2, 8)));
                            continue;
                        }else{
                            bad.push_back(false);
                            BiWeightsVec.push_back(make_pair(startPoint_Weights_Mat, endPoint_weights_Mat));
                        }
                        double theta = rotate_theta[lineNum];

                        // R矩阵
                        MatrixXd R(2, 2);
                        R << cos(theta), -sin(theta), 
                             sin(theta), cos(theta);
                        // ehat
                        MatrixXd ehat(2, 1);
                        ehat << startPoint.col - endPoint.col, startPoint.row - endPoint.row;
                        MatrixXd ehatT_ehat_inverse = (ehat.transpose() * ehat).inverse(); // (ehat^T * ehat)^-1
                        MatrixXd C_line = R * ehat * ehatT_ehat_inverse * ehat.transpose() * R.transpose() - Matrix2d::Identity(2, 2);

                        //线段方向向量e = (start_W_mat - end_W_mat) * Vq
                        MatrixXd C_e = C_line * (startPoint_Weights_Mat - endPoint_weights_Mat);
                        

                        // 按行合并，将C_e 合并到 C_quad_rowStack 下方
                        MatrixXd C_quad_rowStack_tmp(C_quad_rowStack.rows() + C_e.rows(), C_quad_rowStack.cols());
                        C_quad_rowStack_tmp.topRows(C_quad_rowStack.rows()) = C_quad_rowStack;
                        C_quad_rowStack_tmp.bottomRows(C_e.rows()) = C_e;
                        C_quad_rowStack = C_quad_rowStack_tmp;
                    }
                    LineE = SpMat_extendByDiagonal(LineE, C_quad_rowStack, quadIdx, config);
                }
            }
            return LineE;
        }

/**
 * 获取边界能量矩阵参数
 */
void Globalwarp::get_BoundaryE_Matrix(const vector< vector<CoordinateDouble>>& mesh, const Config& config, VectorXd& Boundary, SpMat& Q_boundary) {
    int meshRows = config.meshRows, meshCols = config.meshCols;
    int quadRows = config.quadRows, quadCols = config.quadCols;
    int vertexNum = meshCols * meshRows;

    Boundary = VectorXd::Zero(2 * meshCols * meshRows);
    Q_boundary = SpMat(2 * meshCols * meshRows, 2 * meshCols * meshRows);
    Q_boundary.setZero();

    // left border
    for (int i = 0; i < vertexNum * 2; i += meshCols * 2) {
        Boundary(i) = 0; // 理想边界
        Q_boundary.insert(i, i) = 1;
    }

    // right border
    for (int i = meshCols * 2 - 2; i < vertexNum * 2; i += meshCols * 2) {
        Boundary(i) = config.cols - 1; // 理想边界
        Q_boundary.insert(i, i) = 1;
    }

    // top border
    for (int i = 1; i < 2 * meshCols; i += 2) {
        Boundary(i) = 0; // 理想边界
        Q_boundary.insert(i, i) = 1;
    }

    // bottom border
    for (int i = 2 * vertexNum - 2 * meshCols + 1; i < 2 * vertexNum; i += 2) {
        Boundary(i) = config.rows - 1; // 理想边界
        Q_boundary.insert(i, i) = 1;
    }

    Q_boundary.makeCompressed();
}


/**
 * 获取选择矩阵Q
 */
SpMat Globalwarp::get_SelectMatrix_Q(const vector< vector<CoordinateDouble>>& mesh, const Config& config){
    int meshRows = config.meshRows, meshCols = config.meshCols;
    int quadRows = config.quadRows, quadCols = config.quadCols;

    /**
     * Q = [Qq1 (8 * (2 * meshRows * meshCols))
     *      Qq2
     *      ···
     *      Qqn]
     */
    SpMat SelectMatrix_Q(8 * quadRows * quadCols, 2 * meshRows * meshCols);

    for(int i=0; i<meshRows-1; i++){
        for(int j=0; j<meshCols-1; j++){
            int quadIdx = 8 * (i * quadCols + j); // 找到对应的quad的选择矩阵第一行
            int startIdx_col = 2 * (i * meshCols + j); // 找到当前quad的选择矩阵Qq的开始坐标
            // p0(i, j)
            SelectMatrix_Q.insert(quadIdx, startIdx_col) = 1;
            SelectMatrix_Q.insert(quadIdx+1, startIdx_col+1) = 1;
            // p1(i, j+1)
            SelectMatrix_Q.insert(quadIdx+2, startIdx_col+2) = 1;
            SelectMatrix_Q.insert(quadIdx+3, startIdx_col+3) = 1;
            // p2(i+1, j)
            SelectMatrix_Q.insert(quadIdx+4, startIdx_col + 2 * meshCols) = 1;
            SelectMatrix_Q.insert(quadIdx+5, startIdx_col + 2 * meshCols + 1) = 1;
            // p3(i+1, j+1)
            SelectMatrix_Q.insert(quadIdx+6, startIdx_col + 2 * meshCols + 2) = 1;
            SelectMatrix_Q.insert(quadIdx+7, startIdx_col + 2 * meshCols + 3) = 1;
        }
    }

    SelectMatrix_Q.makeCompressed();
    return SelectMatrix_Q;
}


/**
 * 可视化Straight line
 */
void Globalwarp::Show_StraightLines(cv::Mat src, vector< vector< vector<StraightLine>>>& lineSegmentsInQuad, const Config& config){
    for(int i=0; i<config.quadRows; i++){
        for(int j=0; j<config.quadCols; j++){
            for (int k = 0; k < lineSegmentsInQuad[i][j].size(); k++) {
                StraightLine line = lineSegmentsInQuad[i][j][k];
                draw_StraightLine(src, line);
            }
            if(DEBUG){
                cv::namedWindow("line", cv::WINDOW_AUTOSIZE);
                cv::imshow("line", src);
            }

        }
    }
    if(DEBUG){
        cv::waitKey(0);
    }
}

