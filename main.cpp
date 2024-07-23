#include "config.h"
#include "localwrap.h"
#include "globalwrap.h"

using namespace std;
using namespace cv;

Mat img;
Config config ;
LocalWrap localwrap;
GlobalWrap globalwrap;
vector<vector<CoordinateDouble>> outputmesh;
vector<vector<CoordinateDouble>> mesh;

bool flag_display = true;
double scale_factor = 1;


// 纹理贴图相关
GLuint matToTexture(cv::Mat& mat, GLenum minFilter = GL_LINEAR, GLenum magFilter = GL_LINEAR, GLenum wrapFilter = GL_CLAMP);
void display();

// 后处理
void post_Process(double& sx_avg, double& sy_avg);
void printProgressBar(int current, int total, int width = 50);


int main(int argc, char* argv[]) {
    string imgPath = "/mnt/e/CmakeDemo/PanoWrapping/data/pano1.jpg";
    // 检查命令行参数
    if (argc >= 2) {
        imgPath = argv[1];
    }
    if (argc == 3) {
        try {
            scale_factor = std::stof(argv[2]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid argument: " << argv[2] << " is not a valid number." << std::endl;
            return 1;
        } catch (const std::out_of_range& e) {
            std::cerr << "Out of range: " << argv[2] << " is out of range." << std::endl;
            return 1;
        }
    }
    // 读取图像
    img = imread(imgPath, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Error: Could not read the image." << endl;
        return -1;
    }
    printf("ImageSize: （%d, %d）\n", img.cols, img.rows);
    struct timeval start, end;
    gettimeofday(&start, nullptr); // 记录开始时间


    cout<<"Init ·····"<<endl;
    cv::Mat src;
    cv::resize(img, src, cv::Size(0, 0), 1/scale_factor, 1/scale_factor);
    config = Config(src.rows, src.cols);
    cv::Mat mask = Init_Mask(src);
    
    cout<<"Start LocalWrap ·····"<<endl;

    // 得到seam carving扭曲后像素位移变化矩阵并放置标准网格网
    cv::Mat src_localwrap = src.clone(); 
    vector< vector<CoordinateInt>> displacementField = localwrap.get_displacementField(src_localwrap, mask);  
    cv::imwrite("../res/src_LocalWrap.png", src_localwrap);
    mesh = localwrap.get_rectangleMesh(src_localwrap, config);
    localwrap.draw_Mesh(src_localwrap, mesh, config);
    cv::imwrite("../res/src_StandardMesh.png", src_localwrap);

    // 得到扭曲back后的网格
    cv::Mat src_WithWrapBackMesh = src.clone();
    localwrap.wrap_Back(mesh, displacementField, config);
    localwrap.draw_Mesh(src_WithWrapBackMesh, mesh, config);
    cv::imwrite("../res/src_WrapBackMesh.png", src_WithWrapBackMesh);
    

    cout<<"Start GlobalWrap ·····"<<endl;
    // Q demensions: 8 * quadNum, 2 * vertexNum
    SpMat Q = globalwrap.get_SelectMatrix_Q(mesh, config);  
    // 获取shape energy的系数矩阵 
    SpMat shapeE = globalwrap.get_ShapeE_Coeff(mesh, config);
    
    // 初始化线段分割
    cv::Mat src_alllines = src.clone();
    vector<double> rotate_theta;
    vector<int> lineIdx_BinIdx;
    vector<double> lineIdx_theta;
    vector< vector< vector<StraightLine>>> lineSegmentsInQuad = globalwrap.init_LineSegments(mesh, mask, src_alllines, config, lineIdx_theta, lineIdx_BinIdx, rotate_theta);

    cv::Mat src_quadlines  = src.clone();
    globalwrap.Show_StraightLines(src_quadlines, lineSegmentsInQuad, config);
    cv::imwrite("../res/src_QuadLines.png", src_quadlines);

    // Boundary energy
    VectorXd Boundary; // 理想边界向量
    SpMat Q_boundary; // 边界坐标选择矩阵
    globalwrap.get_BoundaryE_Matrix(mesh, config, Boundary, Q_boundary);
    outputmesh = vector< vector<CoordinateDouble>>(config.meshRows, vector<CoordinateDouble>(config.meshCols));
    outputmesh = mesh;
    cout<< "Start Optimize Implement ······· " <<endl;
    // 迭代优化
    for(int iter = 0; iter < config.iters; iter++){

        // line energy
        int lineNum;
        vector< pair<MatrixXd, MatrixXd>> BiWeightsVec;
        vector<bool> bad;
        SpMat lineE = globalwrap.get_LineE_Matrix(outputmesh, mask, rotate_theta, lineSegmentsInQuad, BiWeightsVec, 
            bad, lineNum, config);
        
        // printf("LineNum: %d\n", lineNum);
        double quadNum = config.quadRows * config.quadCols;
        double LambdaL = config.LambdaL;
        double LambdaB = config.LambdaB;
        double LambdaS = config.LambdaS;
        SpMat shape = (LambdaS / sqrt(quadNum)) * (shapeE * Q);  // 8 * quadNum, 2 * vertexNum
        SpMat line = sqrt((LambdaL / lineNum)) * (lineE * Q); // 2 * lineNum, 2 * vertexNum
        SpMat boundary = sqrt(LambdaB) * Q_boundary; // 2 * vertexNum, 2 * vertexNum

        // 按行合并三个矩阵
        SpMat shape_line = mergeMatricesByRow(shape, line);
        SpMat shape_line_boundary = mergeMatricesByRow(shape_line, boundary);

        VectorXd Bounary_all = VectorXd::Zero(shape_line_boundary.rows());
        Bounary_all.tail(Boundary.size()) = sqrt(LambdaB) * Boundary;

        // update V
        // 1.
        // SpMat A = shape_line_boundary.transpose() * shape_line_boundary; // 2 * vertexNum, 2 * vertexNum
        // VectorXd b = shape_line_boundary.transpose() * Bounary_all;  // 2 * vertexNum, 1
        // Eigen::SimplicialCholesky<SpMat> solver(A);
        // VectorXd V = solver.solve(b);
        // 2.
        // SpMat s_l_b_T = shape_line_boundary.transpose();
        // MatrixXd tmp = s_l_b_T * shape_line_boundary;
        // VectorXd V = tmp.inverse() * shape_line_boundary.transpose() * Bounary_all;
        // 3.
        // SpMat A = shape_line_boundary.transpose() * shape_line_boundary; // 2 * vertexNum, 2 * vertexNum
        // SpMat A_T = A.transpose();
        // VectorXd b = shape_line_boundary.transpose() * Bounary_all;  // 2 * vertexNum, 1
        // SpMat A_T_A = A_T * A;
        // VectorXd A_T_b = A_T * b;
        // Eigen::SimplicialLDLT<SpMat> solver;
        // solver.compute(A_T_A);
        // VectorXd V = solver.solve(A_T_b);
        // 4.
        SpMat A = shape_line_boundary.transpose() * shape_line_boundary; // 2 * vertexNum, 2 * vertexNum
        VectorXd b = shape_line_boundary.transpose() * Bounary_all;  // 2 * vertexNum, 1
        Eigen::SparseQR<SpMat, Eigen::COLAMDOrdering<int>> solver;
        solver.compute(A);
        VectorXd V = solver.solve(b);


        for (int i = 0; i < config.meshRows; i++)
        {
            for (int j = 0; j < config.meshCols; j++)
            {   
                int curIdx = (i * config.meshCols + j) * 2;
                outputmesh[i][j].row = V(curIdx + 1);
                outputmesh[i][j].col = V(curIdx);
            }
        }

        // update theta
        int lineNum_tmp = -1;
        VectorXd thetaBin = VectorXd::Zero(config.thetaBins);
		VectorXd thetaBinCnt = VectorXd::Zero(config.thetaBins);
        for(int i=0; i<config.quadRows; i++){
            for(int j=0; j<config.quadCols; j++){
                vector<StraightLine> lines = lineSegmentsInQuad[i][j];
                int quadIdx = i*config.quadCols + j;
                if(lines.empty()) continue;

                VectorXd Vq = globalwrap.get_Vq(outputmesh, i, j);
                for(int k = 0; k < lines.size(); k++){
                    lineNum_tmp ++;
                    if(bad[lineNum_tmp]) continue;
                    pair<MatrixXd, MatrixXd> BiWeights_start_end = BiWeightsVec[lineNum_tmp];
                    MatrixXd BiWeights_start = BiWeights_start_end.first;
                    MatrixXd BiWeights_end = BiWeights_start_end.second;

                    Vector2d newStartPoint = BiWeights_start * Vq;
                    Vector2d newEndPoint = BiWeights_end * Vq;
                    double theta = atan((newEndPoint - newStartPoint).y() / (newEndPoint - newStartPoint).x());
                    double theta_change = theta - lineIdx_theta[lineNum_tmp];

                    if(isnan(lineIdx_theta[lineNum_tmp]) || isnan(theta_change)) continue;
                    if(theta_change > (PI / 2)) theta_change = -PI + theta_change;
                    else if(theta_change < (-PI / 2)) theta_change = PI + theta_change;

                    int BinIdx = lineIdx_BinIdx[lineNum_tmp];
                    thetaBin[BinIdx] += theta_change;
                    thetaBinCnt[BinIdx]++;
                }
            }
        }

        // 计算theta均值
        for(int t = 0; t < thetaBin.size(); t++) thetaBin[t] /= thetaBinCnt[t];
        for(int t = 0; t < rotate_theta.size(); t++) rotate_theta[t] = thetaBin[lineIdx_BinIdx[t]];
        // 更新进度条
        printProgressBar(iter + 1, config.iters);
    }
    cout<<endl;

    // 缩放回原图像
    scale_mesh(outputmesh, scale_factor, scale_factor,  config);
    scale_mesh(mesh, scale_factor, scale_factor, config);

    // 后处理
    cout<< "Start Post-Process and Wrap ······" <<endl;
    double sx_avg, sy_avg;
    post_Process(sx_avg, sy_avg);
    cout<<"Col_Scale_Factor: "<< sx_avg <<endl;
    cout<<"Row_Scale_Factor: "<< sy_avg <<endl;

    cv::resize(img, img, cv::Size(0, 0), sx_avg, sy_avg);
    scale_mesh(mesh, sy_avg, sx_avg, config);
    scale_mesh(outputmesh, sy_avg, sx_avg, config);

    cv::Mat src_finalmesh = img.clone();
    localwrap.draw_Mesh(src_finalmesh, outputmesh, config);
    cv::imwrite("../res/src_outputMesh.png", src_finalmesh);

    gettimeofday(&end, nullptr); // 记录结束时间
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0; // 计算时间差
    std::cout << "Elapsed time: " << elapsed_time << "s\n"; // 打印时间差


    //glut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);	
	glutInitWindowSize(img.cols, img.rows);
	glutInitWindowPosition(100, 100);	
	glutCreateWindow("Panoramic_image");
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(&display);  
	glutMainLoop(); 

    cv::destroyAllWindows();
    return 0;
}


GLuint matToTexture(cv::Mat& mat, GLenum minFilter , GLenum magFilter , GLenum wrapFilter){
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, img.data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    return textureID;
}
void display() 
{	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	GLuint texGround = matToTexture(img);
    glViewport(0, 0, (GLsizei)img.cols, (GLsizei)img.rows);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, img.cols, img.rows, 0);

	glEnable(GL_TEXTURE_2D);   
	if (flag_display)
	{
		for (int row = 0; row < config.meshRows; row++)
		{
			for (int col = 0; col < config.meshCols; col++)
			{   
                // 顶点坐标
				CoordinateDouble &coord = outputmesh[row][col];

                // 纹理
				CoordinateDouble &localcoord = mesh[row][col];
				localcoord.row /= img.rows;
				localcoord.col /= img.cols;
			}
		}
		flag_display = false;
	}
    
	for (int i = 0; i < config.quadRows; i++) {
		for (int j = 0; j < config.quadCols; j++) {
			CoordinateDouble local_left_top = mesh[i][j];
			CoordinateDouble local_right_top = mesh[i][j + 1];
			CoordinateDouble local_left_bottom = mesh[i + 1][j];
			CoordinateDouble local_right_bottom = mesh[i + 1][j + 1];

			CoordinateDouble global_left_top = outputmesh[i][j];
			CoordinateDouble global_right_top = outputmesh[i][j + 1];
			CoordinateDouble global_left_bottom = outputmesh[i + 1][j];
			CoordinateDouble global_right_bottom = outputmesh[i + 1][j + 1];
			
			glBegin(GL_QUADS);
			glTexCoord2d(local_left_top.col, local_left_top.row); glVertex2d(global_left_top.col, global_left_top.row);
			glTexCoord2d(local_right_top.col, local_right_top.row); glVertex2d(global_right_top.col,  global_right_top.row);
			glTexCoord2d(local_right_bottom.col, local_right_bottom.row); glVertex2d(global_right_bottom.col,  global_right_bottom.row);
			glTexCoord2d(local_left_bottom.col, local_left_bottom.row);	glVertex2d(global_left_bottom.col,  global_left_bottom.row);		
			glEnd();			
		}
	}
    
	glDisable(GL_TEXTURE_2D);
    // 创建一个Mat对象来存储渲染结果
    // cv::Mat renderedImage(img.rows, img.cols, CV_8UC3);

    // // 从OpenGL帧缓冲区读取像素数据
    // glPixelTransferf(GL_RED_SCALE, 1.0f);
    // glPixelTransferf(GL_GREEN_SCALE, 1.0f);
    // glPixelTransferf(GL_BLUE_SCALE, 1.0f);
    // glReadPixels(0, 0, img.cols, img.rows, GL_BGR, GL_UNSIGNED_BYTE, renderedImage.data);

    // flip(renderedImage, renderedImage, 0);
    // // 保存渲染结果
    // cv::imwrite("../res/src_WrapResult.png", renderedImage);
    
	glutSwapBuffers();
}

void post_Process(double& sx_avg, double& sy_avg){

    double sx = 0.0, sy = 0.0;
    for(int i=0; i<config.quadRows; i++){
        for(int j=0; j<config.quadCols; j++){
            CoordinateDouble p0 = mesh[i][j];//左上
			CoordinateDouble p1 = mesh[i][j + 1];//右上
			CoordinateDouble p2 = mesh[i + 1][j];//左下
			CoordinateDouble p3 = mesh[i + 1][j + 1];//右下
			
			CoordinateDouble p0_out = outputmesh[i][j];//左上
			CoordinateDouble p1_out = outputmesh[i][j + 1];//右上
			CoordinateDouble p2_out = outputmesh[i + 1][j];//左下
			CoordinateDouble p3_out = outputmesh[i + 1][j + 1];//右下

            Mat A = (Mat_<double>(1, 4) << p0.row, p1.row, p2.row, p3.row);
            Mat B = (Mat_<double>(1, 4) << p0_out.row, p1_out.row, p2_out.row, p3_out.row);

            double min_y, max_y, min_out_y, max_out_y;
            minMaxIdx(A, &min_y, &max_y);
            minMaxIdx(B, &min_out_y, &max_out_y);
            sy +=  clampValue((max_out_y - max_y) / (min_out_y - min_y), 1, 1.5) ;


            A = (Mat_<double>(1, 4) << p0.col, p1.col, p2.col, p3.col);
            B = (Mat_<double>(1, 4) << p0_out.col, p1_out.col, p2_out.col, p3_out.col);

            double min_x, max_x, min_out_x, max_out_x;
            minMaxIdx(A, &min_x, &max_x);
            minMaxIdx(B, &min_out_x, &max_out_x);
            sx +=   clampValue((max_out_x - min_out_x) / (max_x - min_x) ,  1, 1.5);
        }
    }
    sx_avg = sx / (config.quadRows * config.quadCols);
    sy_avg = sy / (config.quadRows * config.quadCols);
} 

void printProgressBar(int current, int total, int width) {
    float progress = float(current) / total;
    int pos = width * progress;
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "% \r";
    std::cout.flush();
}


