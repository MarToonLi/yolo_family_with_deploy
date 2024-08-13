#include "mygraphicsview.h"
#include "logger.h"
#include <QWheelEvent>
#include <QScrollBar>

MyGraphicsView::MyGraphicsView(QWidget* parent) :
	m_scene(new QGraphicsScene()),
	m_imageItem(new QGraphicsPixmapItem())
{
	m_scene->addItem(m_imageItem);
	setScene(m_scene);
	setStyleSheet("padding: 0px; border: 0px;");//无边框
	setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	setResizeAnchor(QGraphicsView::AnchorUnderMouse);
	setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);   //隐藏水平条
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);     //隐藏竖条
}

MyGraphicsView::~MyGraphicsView()
{
	m_scene->deleteLater();
	delete m_imageItem;
}

void MyGraphicsView::wheelEvent(QWheelEvent* event)
{
	// 滚轮的滚动量
	QPoint scrollAmount = event->angleDelta();
	// 正值表示滚轮远离使用者放大负值表示朝向使用者缩小
	scrollAmount.y() > 0 ? ZoomIn() : ZoomOut();

}

void MyGraphicsView::ZoomIn()
{
	Zoom(1.05);
}

void MyGraphicsView::ZoomOut()
{
	Zoom(0.9);
}

void MyGraphicsView::Zoom(float scaleFactor)
{
	// 防止过小或过大
	qreal factor = transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width();
	if (factor < 0.07 || factor > 100)
		return;

	scale(scaleFactor, scaleFactor);
}


void MyGraphicsView::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton)
	{
		m_isTranslate = true;
		m_lastMousePos = event->pos();
	}
	else if (event->button() == Qt::RightButton)
	{
		QPointF point = mapToScene(event->pos());
		//只有点击图片时才发送
		if (scene()->itemAt(point, transform()) != NULL)
		{
			//emit m_imageBox->ImageClick(point.x(), point.y());
		}
	}

	QGraphicsView::mousePressEvent(event);
}

void MyGraphicsView::mouseMoveEvent(QMouseEvent* event)
{
	{
		if (m_isTranslate)
		{
			//获取
			QPointF mouseDelta = event->pos() - m_lastMousePos;
			Translate(mouseDelta);
		}
		m_lastMousePos = event->pos();
	}

	QGraphicsView::mouseMoveEvent(event);
}

void MyGraphicsView::mouseReleaseEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton)
		m_isTranslate = false;
	if (event->button() == Qt::MiddleButton)
	{
		//QRectF rectItem = scene()->itemsBoundingRect();
		//fitInView(rectItem, Qt::KeepAspectRatio);

		int tx = static_cast<int>(mapToScene(event->pos()).x());
		int ty = static_cast<int>(mapToScene(event->pos()).y());
		//qDebug() << " tx:" << tx << " ty:" << ty;
		if (tx >= 0 && tx < _mat.cols && ty >= 0 && ty < _mat.rows) {

			//ui->le_mousePosition->setText(QVariant(tx).toString() + "," + QVariant(ty).toString());
			uchar* data = _mat.ptr<uchar>(ty);
			//qDebug() << " b:" << data[tx * 3] << " g:" << data[tx * 3 + 1] << " r:" << data[tx * 3 + 2];
			LOGI("value    b:{},g:{},r:{}", data[tx * 3], data[tx * 3 + 1], data[tx * 3 + 2]);
		}
	}



	QGraphicsView::mouseReleaseEvent(event);
}


void MyGraphicsView::Translate(QPointF delta)
{
	int w = viewport()->rect().width();
	int h = viewport()->rect().height();
	QPoint newCenter(w / 2. - delta.x() + 0.5, h / 2. - delta.y() + 0.5);
	centerOn(mapToScene(newCenter));
}
void MyGraphicsView::SetImage(cv::Mat& image, bool _first)
{
	_mat = image.clone();
	m_imageItem->setPixmap(QPixmap::fromImage(QImage((const uchar*)(_mat.data), _mat.cols, _mat.rows, _mat.cols * _mat.channels(), _mat.channels() == 3 ? QImage::Format_RGB888 : QImage::Format_Indexed8)));

	QPoint newCenter(_mat.cols / 2, _mat.rows / 2);

	if (_first)
	{
		QRectF rectItem = scene()->itemsBoundingRect();

		fitInView(rectItem, Qt::KeepAspectRatio);
		////设置scene中心到图像中点
		centerOn(newCenter);
	};


	show();
}