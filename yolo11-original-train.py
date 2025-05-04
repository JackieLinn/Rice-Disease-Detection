from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/Rice-Disease-Detection/ultralytics/ultralytics/cfg/models/11/yolo11n.yaml')
    model.train(data='/root/autodl-tmp/Rice-Disease-Detection/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=128,
                close_mosaic=20,
                workers=16,
                optimizer='SGD',
                lr0=0.01,
                lrf=0.01,
                device='0',
                project='rice-disease-detection/yolo11-original',
                name='train',
                )
