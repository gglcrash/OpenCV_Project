#ifndef COMMON_H
#define COMMON_H

#include <QtGui>//QImage>
#include <stdint.h>
#include <limits>
#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <highgui.h>

#include <QFile>
#include <QTextStream>
#include <QDateTime>

#define sgn(x) ((x > 0) ? 1 : ((x < 0) ? -1 : 0))
#define max_(a,b) ((a) >= (b) ? (a) : (b))
#define min_(a,b) ((a) <= (b) ? (a) : (b))

cv::Mat QImageToCVMat(const QImage *img_qt);
void convertToGrayscale(QImage& image);

cv::Mat FFT2d(cv::Mat& img, double r);                      // Двумерное Фурье преобразование
void Recomb(cv::Mat &src, cv::Mat &dst);
void ForwardFFT(cv::Mat &Src, cv::Mat *FImg);
void InverseFFT(cv::Mat *FImg, cv::Mat &Dst);
void ForwardFFT_Mag_Phase(cv::Mat &src, cv::Mat &Mag, cv::Mat &Phase);
void InverseFFT_Mag_Phase(cv::Mat &Mag, cv::Mat &Phase, cv::Mat &dst);

#endif
