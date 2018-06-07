// seam_carving.cpp

/*
读入图片，编码为Image矩阵
*/

class Image {
private:
    int xrange, yrange;
    int ** color_map;
public:
    Image();
    ~Image();
};