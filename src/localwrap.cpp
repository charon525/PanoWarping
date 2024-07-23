#include "localwrap.h"
#include "config.h"

#define IS_MISSINGPIXEL(mask, row, col) (mask.at<uchar>(row, col) != 0)

using namespace cv;
using namespace std;

/**
 * 获得能量矩阵
 */
Mat LocalWrap::get_EnergyMap(const Mat& src)
{
    Mat gray;
    cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    int rows = src.rows, cols = src.cols;
    Mat energy_map(rows, cols, CV_64FC1, Scalar(0.0));
    Mat M(rows, cols, CV_64FC1, Scalar(0.0));
    for(int i=1; i<rows; i++){
        for(int j=1; j<cols; j++){
            int up = (i-1)%rows, down = (i+1)%rows;
            int left = (j-1)%cols, right = (j+1)%cols; 

            double mU = M.at<double>(up, j);
            double mL = M.at<double>(up, left);
            double mR = M.at<double>(up, right);

            double cU = std::abs(gray.at<uchar>(i, right) - gray.at<uchar>(i, left));
            double cL = std::abs(gray.at<uchar>(up, j) - gray.at<uchar>(i, left)) + cU;
            double cR = std::abs(gray.at<uchar>(up, j) - gray.at<uchar>(i, right)) + cU;

            std::vector<double> cULR = {cU, cL, cR};
            std::vector<double> mULR = {mU, mL, mR};
            for (int k = 0; k < 3; ++k) {
                mULR[k] += cULR[k];
            }

            int argmin = std::min_element(mULR.begin(), mULR.end()) - mULR.begin();
            M.at<double>(i, j) = mULR[argmin];
            energy_map.at<double>(i, j) = cULR[argmin];
        }
    }
    return energy_map;
}


/**
 * 寻找最长缺失边界，确定子图区域，返回起始和最终坐标
 */
vector<int> LocalWrap::find_Longest_MissingBorder(const Mat& src, const Mat& mask, int& border_type) {
    int maxLength = 0, start = -1, end = -1;
    border_type = left_border;

    vector<int> directions = {top_border, bottom_border, left_border, right_border};

    for (int j = 0; j < directions.size(); j++) {
        bool isCounting = false;
        int curLength = 0, curStart = 0, curEnd = 0, cnts = 0;
        if (directions[j] == top_border || directions[j] == bottom_border) {
            cnts = src.cols;
        } else {
            cnts = src.rows;
        }

        for (int i = 0; i < cnts; i++) {
            int row, col;
            switch (directions[j]) {
                case top_border:
                    row = 0;col = i;break;
                case bottom_border:
                    row = src.rows - 1;col = i;break;
                case left_border:
                    row = i;col = 0;break;
                case right_border:
                    row = i;col = src.cols - 1;break;
            }

            if (!IS_MISSINGPIXEL(mask, row, col)) { // 非缺失像素
                if (isCounting) {
                    if (curLength > maxLength) {
                        maxLength = curLength;
                        start = curStart;
                        end = curEnd;
                        border_type = directions[j];
                    }
                }
                isCounting = false;
                curLength = 0;
            } else {  // 缺失像素
                if (!isCounting) {
                    curStart = i;
                    isCounting = true;
                }
                curEnd = i;
                curLength++;
                if (i == cnts - 1 && curLength > maxLength) {
                    maxLength = curLength;
                    start = curStart;
                    end = curEnd;
                    border_type = directions[j];
                }
            }
        }
    }
    
    return {start, end};
}


/**
 * 寻找seamPath
 */
vector<int> LocalWrap::get_Seam_Path_Vertical(Mat src, Mat mask, int seam_type, vector<int> begin_end){
    int subImgRows_start = begin_end[0], subImgRows_end = begin_end[1];
    // cout<<begin_end[0]<<", "<<begin_end[1]<<endl;
    int subImgCols_start = 0, subImgCols_end = src.cols - 1;
    int range = subImgRows_end - subImgRows_start + 1;

    // cout<<src.cols<<", "<<range<<endl;

    cv::Rect subImgROI(subImgCols_start, subImgRows_start, subImgCols_end - subImgCols_start + 1, subImgRows_end - subImgRows_start + 1);
    cv::Mat subImg = src(subImgROI);
    // cout<<"subImg size: "<< subImg.cols<<", "<< subImg.rows<<endl;
    cv::Mat subMask = mask(subImgROI);
 
    Mat energy_map = get_EnergyMap(subImg);


    for (int i = 0; i < subImg.rows; i++) {
        for (int j = 0; j < subImg.cols; j++) {
            if (subMask.at<uchar>(i, j) == 255) {
                energy_map.at<double>(i, j) = 1e8;
            }
            // else if (mask.at<uchar>(i, j) == 254) {
            //     energy_map.at<double>(i, j) = INF_;
            // }
        }
    }



    for(int row = 1; row < energy_map.rows; row ++){
        for(int col = 0; col < energy_map.cols; col ++){
            // cout<<row<<", "<<col<<endl;
            if(col == 0){ // 上方和右上方
                energy_map.at<double>(row, col) += std::min(energy_map.at<double>(row-1, col), energy_map.at<double>(row-1, col+1));
            }else if(col == energy_map.cols - 1){ // 上方和左上方
                energy_map.at<double>(row, col) += std::min(energy_map.at<double>(row-1, col), energy_map.at<double>(row-1, col-1));
            }else{
                energy_map.at<double>(row, col) += std::min(energy_map.at<double>(row-1, col), 
                    std::min(energy_map.at<double>(row-1, col+1) ,energy_map.at<double>(row-1, col-1)));
            }
            
        }
    }

    vector<int> seamPath = vector<int>(range, -1); 
    double minEnergy = energy_map.at<double>(range-1, subImgCols_start);
    seamPath[range-1] = subImgCols_start;

    for(int col = subImgCols_start + 1; col <= subImgCols_end; col++){
        if(minEnergy > energy_map.at<double>(range-1, col)){
            minEnergy = energy_map.at<double>(range-1, col);
            seamPath[range-1] = col;
        }
    }


    for (int row = range - 2; row >= 0; row--) {
        int prevCol = seamPath[row + 1];
        if (prevCol == subImgCols_start) {  // 左边界
            seamPath[row] = (energy_map.at<double>(row, prevCol) > energy_map.at<double>(row, prevCol + 1)) ? prevCol + 1 : prevCol;
        } else if (prevCol == subImgCols_end) {  // 右边界
            seamPath[row] = (energy_map.at<double>(row, prevCol) > energy_map.at<double>(row, prevCol - 1)) ? prevCol - 1 : prevCol;
        } else {  // 中间
            double min_energy = std::min({energy_map.at<double>(row, prevCol), energy_map.at<double>(row, prevCol + 1), energy_map.at<double>(row, prevCol - 1)});
            if (energy_map.at<double>(row, prevCol) == min_energy) {
                seamPath[row] = prevCol;
            } else if (energy_map.at<double>(row, prevCol + 1) == min_energy) {
                seamPath[row] = prevCol + 1;
            } else {
                seamPath[row] = prevCol - 1;
            }
        }
  
    }

    return seamPath;
}

/**
 * 根据seamPath移动像素
 */
void LocalWrap::shiftPixels_BySeamPath_Vertical(Mat& src, Mat& mask, vector<int> seamPath, int seam_type, bool shift2end, vector<int> begin_end, Mat& src_SeamPaths){

    // std::cout<<"start: "<< begin_end[0] <<", end: "<< begin_end[1] << ", seamPath[0]: " << seamPath[0]<<std::endl;

    int rowBegin = begin_end[0], rowEnd = begin_end[1];
    int rows = src.rows, cols = src.cols;

    for(int row = rowBegin; row <= rowEnd; row ++){
        int subImgRow = row - rowBegin;
        if(! shift2end ){ // 像素左侧移动
            for(int col=0; col < seamPath[subImgRow]; col ++){
                cv::Vec3b pixel = src.at<cv::Vec3b>(row, col+1);
                src.at<cv::Vec3b>(row, col) = pixel;
                src_SeamPaths.at<cv::Vec3b>(row, col) = src_SeamPaths.at<cv::Vec3b>(row, col+1);
                mask.at<uchar>(row, col) = mask.at<uchar>(row, col+1);
            }
        }else{
            for(int col=cols-1; col > seamPath[subImgRow]; col --){
                cv::Vec3b pixel = src.at<cv::Vec3b>(row, col-1);
                src.at<cv::Vec3b>(row, col) = pixel;
                src_SeamPaths.at<cv::Vec3b>(row, col) = src_SeamPaths.at<cv::Vec3b>(row, col-1);
                mask.at<uchar>(row, col) = mask.at<uchar>(row, col-1);
            }
        }

		mask.at<uchar>(row, seamPath[subImgRow]) = 0;
        src_SeamPaths.at<cv::Vec3b>(row, seamPath[subImgRow]) = cv::Vec3b(0, 165, 255);//cv::Vec3b(0, 0, 255);
    }

    

}

/**
 * 更新位移场
 */
void LocalWrap::update_displacementField(vector<vector<CoordinateInt>>& displacementField, vector<vector<CoordinateInt>>& Final_displacementField, 
        int seam_type, vector<int> begin_end, vector<int> seamPath, bool shift2end){
            int rows = displacementField.size(), cols = displacementField[0].size();

            for(int row=0; row < rows; row ++){
                for(int col=0; col < cols; col ++){
                    CoordinateInt coordinate;
                    if(seam_type == seam_vertical && row >= begin_end[0] && row <= begin_end[1]){
                        int subImgRow = row - begin_end[0];
                        if(col > seamPath[subImgRow] && shift2end) coordinate.col = -1;
                        else if(col < seamPath[subImgRow] && !shift2end) coordinate.col = 1;
                    }else if(seam_type == seam_horizontal && col >= begin_end[0] && col <= begin_end[1]){
                        int subImgCol = col - begin_end[0];
                        if(row > seamPath[subImgCol] && shift2end) coordinate.row = -1;
                        else if(row < seamPath[subImgCol] && !shift2end) coordinate.row = 1;
                    }

                    
                    // 计算像素原始位置
                    int row_tmp = row + coordinate.row;
                    int col_tmp = col + coordinate.col;
                    CoordinateInt coor_target = displacementField[row_tmp][col_tmp];
                    int rowOrigin = row_tmp + coor_target.row;
                    int colOrigin = col_tmp + coor_target.col;
                    
                    // 更新位移
                    CoordinateInt& coor_final = Final_displacementField[row][col];
                    coor_final.row = rowOrigin - row;
                    coor_final.col = colOrigin - col;  
                }
            }
            displacementField = Final_displacementField;
        }

/**
 * 获取位移场displacementField
 */
vector<vector<CoordinateInt>> LocalWrap::get_displacementField(Mat& src, Mat& mask) {
    int rows = src.rows, cols = src.cols;

    vector<vector<CoordinateInt>> displacementField(rows, vector<CoordinateInt>(cols, CoordinateInt(0, 0)));
    vector<vector<CoordinateInt>> Final_displacementField(rows, vector<CoordinateInt>(cols, CoordinateInt(0, 0)));

    cv::namedWindow("LocalWrap Process Result", cv::WINDOW_NORMAL);
    
    Mat src_SeamPath = src.clone();
    bool processing = true;
    while (processing) {
        int border_type;
        vector<int> begin_end = find_Longest_MissingBorder(src, mask, border_type);
        bool shift2end = (border_type == bottom_border || border_type == right_border);
        if (begin_end[0] == begin_end[1] && begin_end[0] < 0) {
            processing = false; // 填充完毕
        } else {
            int seam_type = (border_type == left_border || border_type == right_border) ? seam_vertical : seam_horizontal;
            vector<int> seamPath;
            if(seam_type == seam_horizontal){
                seamPath = get_Seam_Path_Horizontal(src, mask, seam_type, begin_end);
                shiftPixels_BySeamPath_Horizontal(src, mask, seamPath, seam_type, shift2end, begin_end, src_SeamPath);
            }else{
                seamPath = get_Seam_Path_Vertical(src, mask, seam_type, begin_end);
                shiftPixels_BySeamPath_Vertical(src, mask, seamPath, seam_type, shift2end, begin_end, src_SeamPath);
            }
            update_displacementField(displacementField, Final_displacementField, seam_type, begin_end, seamPath, shift2end);
        }

        
        // 确保src和mask具有相同的类型和通道数
        Mat src_display, mask_display;
        if (src.channels() == 1) {
            cvtColor(src, src_display, cv::COLOR_GRAY2BGR);
        } else {
            src.copyTo(src_display);
        }
        if (mask.channels() == 1) {
            cvtColor(mask, mask_display, cv::COLOR_GRAY2BGR);
        } else {
            mask.copyTo(mask_display);
        }
        // 确保两个图像具有相同的大小
        cv::resize(mask_display, mask_display, src_display.size());
        // 将src_display和mask_display水平连接
        Mat combined;
        hconcat(src_display, mask_display, combined);
        // 显示合并后的图像
        cv::imshow("LocalWrap Process Result", combined);
        if(DEBUG)
            cv::waitKey(1);
    }
    cv::imwrite("../res/src_SeamPaths.png", src_SeamPath);
    if(DEBUG){
        cv::waitKey(0);
    }
    
    return displacementField;
}


/**
 * 放置矩形网格网
 */

vector< vector<CoordinateDouble>> LocalWrap::get_rectangleMesh(Mat& src, Config config){
    int rows = config.rows, cols =  config.cols; // 图片行数， 列数
    int meshRows = config.meshRows, meshCols = config.meshCols; // 网格行数，列数
    double meshRowSize = config.meshRowSize, meshColSize = config.meshColSize; // 网格跨度(跨过的像素数目)

    vector< vector<CoordinateDouble>> mesh = vector< vector<CoordinateDouble>>(meshRows, 
                                vector<CoordinateDouble>(meshCols, CoordinateDouble(0.0, 0.0)));
    for(int i=0; i<meshRows; i++){
        for(int j=0; j<meshCols; j++){
            mesh[i][j].row = i * meshRowSize;
            mesh[i][j].col = j * meshColSize;
        }
    }
    return mesh;
}

/**
 * 扭曲回原图像
 */
void LocalWrap::wrap_Back(vector<vector<CoordinateDouble>>& mesh, const vector<vector<CoordinateInt>>& displacementField, const Config& config) {
    const int meshrows = config.meshRows;
    const int meshcols = config.meshCols;

    for (int row = 0; row < meshcols; ++row) {
        for (int col = 0; col < meshcols; ++col) {
            CoordinateDouble& meshVertexPoint = mesh[row][col];
            
            int displ_row = static_cast<int>(floor(meshVertexPoint.row));
            int displ_col = static_cast<int>(floor(meshVertexPoint.col));
            
            // 处理边缘情况
            if (row == meshrows - 1 && col == meshcols - 1) {
                --displ_row;
                --displ_col;
            }
            
            // 确保索引在有效范围内
            displ_row = std::max(0, std::min(displ_row, static_cast<int>(displacementField.size()) - 1));
            displ_col = std::max(0, std::min(displ_col, static_cast<int>(displacementField[0].size()) - 1));
            
            const CoordinateInt& vertexDisplacement = displacementField[displ_row][displ_col];
            
            meshVertexPoint.row += vertexDisplacement.row;
            meshVertexPoint.col += vertexDisplacement.col;
        }
    }
}

/**
 * 绘制网格网
 */
void LocalWrap::draw_Mesh(Mat& src, vector< vector<CoordinateDouble>>& mesh, Config config)
{
    int meshRows = config.meshRows, meshCols = config.meshCols;
    cv::namedWindow("Src_with_Mesh", cv::WINDOW_NORMAL);

    for (int i = 0; i < meshRows; i++)
    {
        for (int j = 0; j < meshCols; j++)
        {
            CoordinateDouble cur_meshVertex = mesh[i][j];
            if(i == meshRows-1 && j < meshCols-1){ // 最后一行,画当前点到右侧点的直线
                CoordinateDouble right_meshVertex = mesh[i][j+1];
                draw_StraightLine(src, StraightLine(cur_meshVertex, right_meshVertex));
            }else if(i < meshRows-1 && j == meshCols-1){ // 最后一列,画当前点到下侧点的直线
                CoordinateDouble bottom_meshVertex = mesh[i+1][j];
                draw_StraightLine(src, StraightLine(cur_meshVertex, bottom_meshVertex));
            }else if(i < meshRows-1 && j < meshCols-1){
                CoordinateDouble right_meshVertex = mesh[i][j+1];
                draw_StraightLine(src, StraightLine(cur_meshVertex, right_meshVertex));
                CoordinateDouble bottom_meshVertex = mesh[i+1][j];
                draw_StraightLine(src, StraightLine(cur_meshVertex, bottom_meshVertex));
            }
            cv::imshow("Src_with_Mesh", src);
        }
    }
    if(DEBUG) {
        cv::waitKey(0);
    }
}


/**
 * 寻找Horizontal seamPath
 */
vector<int> LocalWrap::get_Seam_Path_Horizontal(Mat src, Mat mask, int seam_type, vector<int> begin_end){
    // cout<<"get_Seam_Path_Horizontal······"<<endl;
    int subImgRows_start = 0, subImgRows_end = src.rows - 1;
    int subImgCols_start = begin_end[0], subImgCols_end = begin_end[1];
    int range = subImgCols_end - subImgCols_start + 1;

    cv::Rect subImgROI(subImgCols_start, subImgRows_start, subImgCols_end - subImgCols_start + 1, subImgRows_end - subImgRows_start + 1);
    cv::Mat subImg = src(subImgROI);
    cv::Mat subMask = mask(subImgROI);
    
    

    Mat Energy_map, from;
    Energy_map = get_EnergyMap(subImg);
    from.create(subImg.size(), CV_32SC1); // sub-image

    for (int i = 0; i < subImg.rows; i++) {
        for (int j = 0; j < subImg.cols; j++) {
            if (subMask.at<uchar>(i, j) == 255) {
                Energy_map.at<double>(i, j) = 1e8;
            }
        }
    }


    for(int j=1; j<=subImg.cols-1; j++){
        for(int i=0; i<=subImg.rows-1; i++){
            if(i == 0){ // 左和左下
                if(Energy_map.at<double>(i, j-1) > Energy_map.at<double>(i+1, j-1)){
                    Energy_map.at<double>(i, j) += Energy_map.at<double>(i+1, j-1);
                    from.at<int>(i, j) = i + 1;
                }else{
                    Energy_map.at<double>(i, j) += Energy_map.at<double>(i, j-1);
                    from.at<int>(i, j) = i;
                }
            }else if(i == subImg.rows-1){ // 左和左上
                if(Energy_map.at<double>(i, j-1) >  Energy_map.at<double>(i-1, j-1)){
                    Energy_map.at<double>(i, j) += Energy_map.at<double>(i-1, j-1);
                    from.at<int>(i, j) = i - 1;
                }else{
                    Energy_map.at<double>(i, j) += Energy_map.at<double>(i, j-1);
                    from.at<int>(i, j) = i;
                }
            }else{
                double min_e = min({Energy_map.at<double>(i+1, j-1), Energy_map.at<double>(i, j-1), Energy_map.at<double>(i-1, j-1)});
                if(min_e == Energy_map.at<double>(i+1, j-1)){
                    Energy_map.at<double>(i, j) += Energy_map.at<double>(i+1, j-1);
                    from.at<int>(i, j) = i + 1;
                }else if(min_e == Energy_map.at<double>(i, j-1)){
                    Energy_map.at<double>(i, j) += Energy_map.at<double>(i, j-1);
                    from.at<int>(i, j) = i;
                }else{
                    Energy_map.at<double>(i, j) += Energy_map.at<double>(i-1, j-1);
                    from.at<int>(i, j) = i - 1;
                }
            }
        }
    }

    vector<int> seamPath = vector<int>(range, -1); 
    double minEnergy = Energy_map.at<double>(subImgRows_start, range-1);
    seamPath[range-1] = subImgRows_start;

    for(int i=subImgRows_start+1; i<=subImgRows_end; i++){
        if(Energy_map.at<double>(i, range-1) < minEnergy){
            minEnergy = Energy_map.at<double>(i, range-1);
            seamPath[range-1] = i;
        }
    }

    for(int j = range-2; j >= 0; j--){
        int preRow = seamPath[j+1];
        seamPath[j] = from.at<int>(seamPath[j+1], j+1);
    }
    return seamPath;
}

void LocalWrap::shiftPixels_BySeamPath_Horizontal(Mat& src, Mat& mask, vector<int> seamPath, int seam_type, bool shift2end, vector<int> begin_end, Mat& src_SeamPaths){
    int colBegin = begin_end[0], colEnd = begin_end[1];
    int rows = src.rows, cols = src.cols;

    for(int j = colBegin; j<=colEnd; j++){
        int subImgcol = j - colBegin;
        if(!shift2end){ // 像素向上移动
            for(int i=0; i<seamPath[subImgcol]; i++){
                cv::Vec3b pixel = src.at<cv::Vec3b>(i+1, j);
                src.at<cv::Vec3b>(i, j) = pixel;
                src_SeamPaths.at<cv::Vec3b>(i, j) = src_SeamPaths.at<cv::Vec3b>(i+1, j);
                mask.at<uchar>(i, j) = mask.at<uchar>(i+1, j);
            }
        }else{
            for(int i=rows-1; i>seamPath[subImgcol]; i--){
                cv::Vec3b pixel = src.at<cv::Vec3b>(i-1, j);
                src.at<cv::Vec3b>(i, j) = pixel;
                src_SeamPaths.at<cv::Vec3b>(i, j) = src_SeamPaths.at<cv::Vec3b>(i-1, j);
                mask.at<uchar>(i, j) = mask.at<uchar>(i-1, j);
            }
        }

        mask.at<uchar>(seamPath[subImgcol], j) = 0;
        src_SeamPaths.at<cv::Vec3b>(seamPath[subImgcol], j) = cv::Vec3b(0, 165, 255);//cv::Vec3b(0, 0, 255);
    }

}