#-------------------------------------------------
#
# Project created by QtCreator 2017-04-26T11:14:47
#
#-------------------------------------------------

TEMPLATE = app
TARGET = fft_wm

QT       += core
QT       += gui
CONFIG   += console
CONFIG   -= app_bundle

HEADERS = src\common.h \
    src/main.h

SOURCES = src\main.cpp \
          src\common.cpp

INCLUDEPATH += $$quote(c:/opencv/2.3/build/include) \
$$quote(c:/opencv/2.3/build/include/opencv)

LIBS += $$quote(c:/opencv/2.3/build/x86/mingw/lib/*.a)

CONFIG(debug, debug|release) {
    LIBS += -Llib #-...
} else {
    LIBS += -Llib #-...
}


