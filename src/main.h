#ifndef MAIN_H
#define MAIN_H

#include <QtCore/QCoreApplication>
#include <stdint.h>
#include <limits>
#include <fstream>

void embedWmToFFT(const QString fileCont, const QString fileWm, QString fileRes);
void extractWmFromFFT(const QString fileCont);
cv::Mat extractWmFromFFT2(cv::Mat blueSrc);

void embedWmToDCT(const QString fileCont, const QString fileWm, QString fileRes);
void extractWmFromDCT(const QString fileCont);

void rotateImg(const QString fileCont);
void testImageRotation(const QString fileCont);
int getAngleValue(cv::Mat src, cv::Mat idealSign);

#endif // MAIN_H
