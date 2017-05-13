#include <QtCore/QCoreApplication>
#include <stdint.h>
#include <limits>
#include "common.h"

void embedWmToFFT(const QString fileCont, const QString fileWm, QString fileRes);
void extractWmFromFFT(const QString fileCont);
void extractWmFromFFT(cv::Mat blueSrc);

void embedWmToDCT(const QString fileCont, const QString fileWm, QString fileRes);
void extractWmFromDCT(const QString fileCont);

void rotateImg(const QString fileCont);


int main(int argc, char *argv[])
{
     QCoreApplication a(argc, argv);
 QString strCont = "D:\\opencvimg\\Lenna.png";
QString strSign = "D:\\opencvimg\\sign00003.png";
QString strRes = "D:\\opencvimg\\imgDst40.png";

 //embedWmToFFT(strCont, strSign, strRes);
 //extractWmFromFFT(strRes);
rotateImg(strRes);

    if(argc == 4)
    {
        QString fileCont = argv[1],
                fileWm   = argv[2],
                fileRes  = argv[3];
        //embedWmToFFT(fileCont, fileWm, fileRes);
        embedWmToDCT(fileCont, fileWm, fileRes);
    }
    else if(argc == 3)
    {
        QString fileCont = argv[1],
                fileMag  = argv[2];
        rotateImg(fileCont);
    }
    else if(argc == 2)
    {
        QString fileCont = argv[1];
        //extractWmFromFFT(fileCont);
        extractWmFromDCT(fileCont);
    }
    return a.exec();
}

void embedWmToFFT(const QString fileCont, const QString fileWm, QString fileRes)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1), imgDst32, imgDst,
            imgWm = cv::imread(fileWm.toStdString().c_str(), 0), imgWm32,
            imgMag, imgPhase;     // Ампльтуда спектра и Фаза спектра
    imgWm.convertTo(imgWm32, CV_32F, 1.0/255.0);

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);

    // Раскладываем изображение в спектр
    ForwardFFT_Mag_Phase(imgSrcCh[2], imgMag, imgPhase);
    cv::add(imgMag, imgWm32, imgMag);

    // Обратное преобразование
    InverseFFT_Mag_Phase(imgMag, imgPhase, imgDst32);

    double minVal, maxVal;
    cv::minMaxLoc(imgDst32, &minVal, &maxVal);       // минимальное и максимальное значение яркости
    imgDst32 *= maxVal;
    imgDst32.convertTo(imgDst, CV_8U, 255.0/(maxVal-minVal), -minVal);
    cv::normalize(imgDst, imgDst, 0, 255, cv::NORM_MINMAX);

    imgSrcCh[2] = imgDst.clone();
    cv::merge(imgSrcCh, imgDst);

    // Вывод результатов
    cv::Mat imgLogMag;
    imgLogMag.zeros(imgMag.rows, imgMag.cols, CV_32F);
    imgLogMag=(imgMag+1);
    cv::log(imgLogMag, imgLogMag);
    //---------------------------------------------------
    cv::imshow("Логарифм амплитуды", imgLogMag);
    cv::imshow("Фаза", imgPhase);
    cv::imwrite(fileRes.toStdString().c_str(), imgDst);
}

void extractWmFromFFT(const QString fileCont)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1),
            imgMag, imgPhase;

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);
    // Раскладываем изображение в спектр
    ForwardFFT_Mag_Phase(imgSrcCh[2], imgMag, imgPhase);

    cv::Mat imgLogMag;
    imgLogMag.zeros(imgMag.rows, imgMag.cols, CV_32F);
    imgLogMag = (imgMag+1);
    cv::log(imgLogMag, imgLogMag);
    cv::imshow("Логарифм амплитуды 1", imgLogMag);
}

void extractWmFromFFT(cv::Mat blueSrc)
{
    cv::Mat imgMag, imgPhase;
    // Раскладываем изображение в спектр
    ForwardFFT_Mag_Phase(blueSrc, imgMag, imgPhase);

    cv::Mat imgLogMag;
    imgLogMag.zeros(imgMag.rows, imgMag.cols, CV_32F);
    imgLogMag = (imgMag+1);
    cv::log(imgLogMag, imgLogMag);
    imgLogMag.convertTo(imgLogMag,-1,10,0);

    cv::Rect myROI(0.78125*imgLogMag.cols, 0.78125*imgLogMag.rows, 0.21875*imgLogMag.cols, 0.21875*imgLogMag.rows);
    cv::Mat croppedImage = imgLogMag(myROI);

    cv::normalize(croppedImage, croppedImage, 0, 255, cv::NORM_MINMAX);
    cv::threshold(croppedImage, croppedImage, 70, 255,CV_THRESH_BINARY);
    cv::imshow("Cropped Sign", croppedImage);

}


void embedWmToDCT(const QString fileCont, const QString fileWm, QString fileRes)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1), imgSrc32, imgDst32, imgDst,
            imgWm = cv::imread(fileWm.toStdString().c_str(), 0), imgWm32,
            imgMag32, imgMag;     // Ампльтуда спектра

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);

    /*imgSrc*/imgSrcCh[0].convertTo(imgSrc32, CV_32F, 1.0/255.0);
    imgWm.convertTo(imgWm32, CV_32F, 1.0/255.0);
    imgWm32 /= 5;

    // Раскладываем изображение в спектр
    cv::dct(imgSrc32, imgMag32, 0);
    cv::add(imgMag32, imgWm32, imgMag32);

    // Обратное DCT и собинаем в изображение
    cv::idct(imgMag32, imgDst32);
    imgDst32.convertTo(imgDst, CV_8U, 255.0, 0);
    cv::normalize(imgDst, imgDst, 0, 255, cv::NORM_MINMAX);
    cv::imwrite(fileRes.toStdString().c_str(), imgDst);

    double minVal, maxVal;
    minMaxLoc(imgMag32, &minVal, &maxVal);       // минимальное и максимальное значение яркости
    imgMag32 = imgMag32*maxVal;
    imgMag32.convertTo(imgMag, CV_8U, 255.0/(maxVal-minVal), -minVal);
    cv::normalize(imgMag, imgMag, 0, 255, cv::NORM_MINMAX);
    cv::imshow("Mag_DST8", imgMag);
}


void extractWmFromDCT(const QString fileCont)
{
    cv::Mat imgSrc = cv::imread(fileCont.toStdString().c_str(), 1/*1??*/),
            imgSrc32, imgMag32, imgMag;     // Ампльтуда спектра

    // Разобъем изображение по каналам
    std::vector <cv::Mat> imgSrcCh;
    cv::split(imgSrc, imgSrcCh);

    /*imgSrc*/imgSrcCh[0].convertTo(imgSrc32, CV_32F, 1.0/255.0);
    cv::dct(imgSrc32, imgMag32,0);

    double minVal, maxVal;
    minMaxLoc(imgMag32, &minVal, &maxVal);       // минимальное и максимальное значение яркости
    imgMag32 = imgMag32*maxVal;
    imgMag32.convertTo(imgMag, CV_8U, 255.0/(maxVal-minVal), -minVal);
    cv::normalize(imgMag, imgMag, 0, 255, cv::NORM_MINMAX);
    cv::imshow("img Mag", imgMag);
}


void rotateImg(const QString fileCont)
{
    cv::Mat src = cv::imread(fileCont.toStdString().c_str(), 1);
    std::vector <cv::Mat> imgSrcCh;
    cv::split(src, imgSrcCh);
    cv::Mat blueSrc = imgSrcCh[2].clone();
    cv::Mat dst;

    cv::Point2f pc(src.cols/2., src.rows/2.);
    cv::Mat r = cv::getRotationMatrix2D(pc, 75, 1.0);
    cv::warpAffine(blueSrc, dst, r, src.size());

    extractWmFromFFT(dst);


   // cv::imshow("imgMagRes", dst);
}
