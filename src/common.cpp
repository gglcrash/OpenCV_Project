#include "common.h"

// ------------------------------------------------------------------
// Преобразование QImage в cv::Mat and
// ------------------------------------------------------------------
cv::Mat QImageToCVMat(const QImage *img_qt)
{
    if(img_qt->isNull()) return cv::Mat(0,0,CV_8U);

    if(img_qt->depth() == 32)
    {
        cv::Mat img_mat = cv::Mat(img_qt->height(), img_qt->width(),
                                     CV_8UC4, (uchar*)img_qt->bits(), img_qt->bytesPerLine());
        cv::Mat img_cv  = cv::Mat(img_mat.rows, img_mat.cols, CV_8UC3);
        int from_to[] = {0,0,  1,1,  2,2};
        cv::mixChannels(&img_mat, 1, &img_cv, 1, from_to, 3);
        return img_cv;
    }
    else if(img_qt->depth() == 8)
    {
        cv::Mat img_mat = cv::Mat(img_qt->height(), img_qt->width(),
                                  CV_8U, (uchar*)img_qt->bits(), img_qt->bytesPerLine());

        cv::Mat img_mat1 = cv::Mat(img_mat.rows, img_mat.cols, CV_8U);
        img_mat1 = img_mat.clone();

        return img_mat1;
    }
    else
        return cv::Mat(0,0,CV_8U);
}


void convertToGrayscale(QImage& image)
{
    if(image.width() == 0 || image.height() == 0 ||
       (image.depth()!=32 && image.depth()!=24)) return;

    unsigned int imwidth = image.width(),
                 imheight = image.height();

    QImage imgres(imwidth, imheight, QImage::Format_Indexed8);
    for(int i=0; i<256; imgres.setColor(i,qRgb(i,i,i)), i++);

    for(unsigned int i=0; i<imheight; i++)
    {
        QRgb *line = (QRgb*)image.scanLine(i);
        for(unsigned int j=0; j<imwidth; j++)
        {
            imgres.setPixel(j,i,qGray(*(line+j)));
        }
    }

    image = imgres.copy(imgres.rect());
}


//----------------------------------------------------------
// Двумерное Фурье преобразование
//----------------------------------------------------------
cv::Mat FFT2d(cv::Mat& img, double r)
{
    const double E=2.7182818284590452353602874713527;

    // *Для ускорения обработки уменьшим размер изображения
    cv::Mat img_small = img.clone();
    if(img.cols >= 2000)
    cv::resize(img, img_small, cv::Size(1000,1000));

    int w=img_small.cols,
        h=img_small.rows;

    // Производим свертку
    cv::Mat Mag,           // Амплитуда спектра
            Phase;         // Фаза спектра
    ForwardFFT_Mag_Phase(img_small, Mag, Phase);

    // Создаем частотный фильтр
    cv::Mat filter = cv::Mat::zeros(Mag.rows, Mag.cols, CV_32F);
    for(int i=0; i < h; i++)
    {
        double I = (float)i-((float)h-1)/2.0; //2
        float* F = filter.ptr<float>(i);
        for(int j=0; j < w; j++)
        {
            double J = (float)j-((float)w-1)/2.0; //2
            double f = sqrtl(I*I+J*J);
            F[j] = f*powf(E,-powf((f/r),6.0));//4
        }
    }
    cv::multiply(Mag, filter, Mag);
    //cv::multiply(Phase, filter, Phase);

    ///cv::Mat img_gaus;
    ///cv::GaussianBlur(Phase, img_gaus, cv::Size(0,0), 3); //0,0
    ///cv::addWeighted(Phase, 2.0, img_gaus, -1.0, 0, Phase);



    // Обратное преобразование
    cv::Mat img_res;
    InverseFFT_Mag_Phase(Mag, Phase, img_res);
    img_res = img_res(cv::Range(0, h), cv::Range(0, w));

    double minVal, maxVal;
    minMaxLoc(img_res, &minVal, &maxVal);       // минимальное и максимальное значение яркости
    img_res = img_res*maxVal;

    cv::Mat img8;
    img_res.convertTo(img8, CV_8U, 255.0/(maxVal-minVal), -minVal);
    cv::normalize(img8, img8, 0, 255, cv::NORM_MINMAX);

    cv::Mat img_b;
    cv::resize(img8, img_b, img.size());

    return img_b;   //8
}

//----------------------------------------------------------
// Функция для перестановки четвертей изображения местамии
// так, чтобы ноль спектра находился в центре изображения.
//----------------------------------------------------------
void Recomb(cv::Mat &src, cv::Mat &dst)
{
    int cx = src.cols >> 1;
    int cy = src.rows >> 1;

    cv::Mat q0(src, cv::Rect(0, 0, cx, cy));   // Top-Left
    cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(src, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(src, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;        // swap (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);     // swap (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    dst = src;
}

//----------------------------------------------------------
// По заданному изображению рассчитывает
// действительную и мнимую части спектра Фурье
//----------------------------------------------------------
void ForwardFFT(cv::Mat &Src, cv::Mat *FImg)
{
    int M = cv::getOptimalDFTSize(Src.rows);
    int N = cv::getOptimalDFTSize(Src.cols);
    cv::Mat padded;

    cv::copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Создаем комплексное представление изображения
    // planes[0] содержит само изображение, planes[1] его мнимую часть (заполнено нулями)
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg);

    // После преобразования результат так-же состоит из действительной и мнимой части
    cv::split(complexImg, planes);

    // Обрежем спектр, если у него нечетное количество строк или столбцов
    planes[0] = planes[0](cv::Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
    planes[1] = planes[1](cv::Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));

    Recomb(planes[0],planes[0]); Recomb(planes[1],planes[1]);

    // Нормализуем спектр
    planes[0]/=float(M*N);      planes[1]/=float(M*N);
    FImg[0]=planes[0].clone();  FImg[1]=planes[1].clone();
}


//----------------------------------------------------------
// По заданным действительной и мнимой части
// спектра Фурье восстанавливает изображение
//----------------------------------------------------------
void InverseFFT(cv::Mat *FImg, cv::Mat &Dst)
{
    Recomb(FImg[0],FImg[0]);
    Recomb(FImg[1],FImg[1]);
    cv::Mat complexImg;
    cv::merge(FImg, 2, complexImg);

    // Производим обратное преобразование Фурье
    cv::idft(complexImg, complexImg);
    cv::split(complexImg, FImg);
    cv::normalize(FImg[0], Dst, 0, 1, CV_MINMAX);
}


//----------------------------------------------------------
// Раскладывает изображение на амплитуду и фазу спектра Фурье
//----------------------------------------------------------
void ForwardFFT_Mag_Phase(cv::Mat &src, cv::Mat &Mag, cv::Mat &Phase)
{
    cv::Mat planes[2];
    ForwardFFT(src,planes);
    Mag.zeros(planes[0].rows, planes[0].cols, CV_32F);
    Phase.zeros(planes[0].rows, planes[0].cols, CV_32F);
    cv::cartToPolar(planes[0], planes[1], Mag,Phase);
}

//----------------------------------------------------------
// По заданным амплитуде и фазе
// спектра Фурье восстанавливает изображение
//----------------------------------------------------------
void InverseFFT_Mag_Phase(cv::Mat &Mag, cv::Mat &Phase, cv::Mat &dst)
{
    cv::Mat planes[2];
    planes[0].create(Mag.rows, Mag.cols, CV_32F);
    planes[1].create(Mag.rows, Mag.cols, CV_32F);
    cv::polarToCart(Mag, Phase, planes[0], planes[1]);
    InverseFFT(planes, dst);
}
