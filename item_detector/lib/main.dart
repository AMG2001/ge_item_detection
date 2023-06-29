import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:image/image.dart' as imglib;
import 'package:tflite_flutter/tflite_flutter.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  cameras = await availableCameras();

  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.purple,
      ),
      home: ObjectDetectionView(),
    );
  }
}

class ObjectDetectionView extends StatefulWidget {
  @override
  _ObjectDetectionViewState createState() => _ObjectDetectionViewState();
}

class _ObjectDetectionViewState extends State<ObjectDetectionView> {
  late CameraController _cameraController;
  late List<CameraDescription> _cameras;
  bool _isCameraInitialized = false;
  Interpreter? _interpreter;
  List<dynamic>? _recognitions;
  int _imageWidth = 0;
  int _imageHeight = 0;
  double _cameraAspectRatio = 1.0;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadModel();
  }

  @override
  void dispose() {
    _cameraController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Object Detection with TFLite'),
      ),
      body: Stack(
        children: [
          _cameraPreviewWidget(),
          _bboxesWidget(),
        ],
      ),
    );
  }

  Widget _cameraPreviewWidget() {
    if (!_isCameraInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return AspectRatio(
      aspectRatio: _cameraAspectRatio,
      child: CameraPreview(_cameraController),
    );
  }

  Widget _bboxesWidget() {
    return Positioned.fill(
      child: CustomPaint(
        painter: BoundingBoxPainter(_recognitions, _imageHeight, _imageWidth),
      ),
    );
  }

  Future<void> _initializeCamera() async {
    _cameras = await availableCameras();
    _cameraController =
        CameraController(_cameras[0], ResolutionPreset.veryHigh);
    _cameraController.initialize().then((_) {
      if (!mounted) {
        return;
      }
      setState(() {
        _cameraAspectRatio = _cameraController.value.aspectRatio;
        _isCameraInitialized = true;
      });
      _cameraController.startImageStream((image) => _processCameraImage(image));
    });
  }

  void _processCameraImage(CameraImage image) async {
    if (_interpreter == null) return;
    imglib.Image? convertedImage =
        _resizeImage(_convertYUV420toImageColor(image)!, 300);
    if (convertedImage == null) return;

    var inputTensor = _interpreter!.getInputTensor(0);
    var inputShape = inputTensor.shape;
    var inputImage = imageToByteListFloat32(
        convertedImage, inputShape[1], inputShape[2], 127.5, 127.5);

    // Get output shape and allocate output buffer
    var outputTensor = _interpreter!.getOutputTensor(0);
    var outputShape = outputTensor.shape;
    var outputData = List.filled(outputShape.reduce((a, b) => a * b), 0.0);

    // Use the run method with allocated output buffer
    _interpreter!.run(inputImage, outputData);

    setState(() {
      _recognitions = outputData;
      _imageWidth = convertedImage.width;
      _imageHeight = convertedImage.height;
    });
  }

  Uint8List imageToByteListFloat32(imglib.Image image, int inputSize,
      int outputSize, double mean, double std) {
    var convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (var i = 0; i < inputSize; i++) {
      for (var j = 0; j < inputSize; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imglib.getRed(pixel) - mean) / std;
        buffer[pixelIndex++] = (imglib.getGreen(pixel) - mean) / std;
        buffer[pixelIndex++] = (imglib.getBlue(pixel) - mean) / std;
      }
    }
    return convertedBytes.buffer.asUint8List();
  }

  imglib.Image? _convertYUV420toImageColor(CameraImage image) {
  try {
    final imglib.Image resultImage = imglib.Image(image.width, image.height);
    int planeIndex = 0;
    for (final plane in image.planes) {
      final bytes = plane.bytes;
      int row = 0;
      for (int y = 0; y < image.height; y++) {
        int col = 0;
        for (int x = 0; x < image.width; x++) {
          resultImage.setPixel(x, y, imglib.getColor(bytes[row + col], 0, 0));
          col += plane.bytesPerRow;
        }
        row += (plane.bytesPerRow ?? 1) * (plane.height ?? 1);
      }
      planeIndex++;
    }
    return resultImage;
  } catch (e) {
    print('Error converting YUV420 image to RGB: $e');
    return null;
  } 
}

  imglib.Image? _resizeImage(imglib.Image src, int newSize) {
    try {
      return imglib.copyResize(src, width: newSize, height: newSize);
    } catch (e) {
      print('Error resizing image: $e');
      return null;
    }
  }

  Future<void> _loadModel() async {
    try {
      // final gpuDelegate = GpuDelegate();
      final interpreterOptions = InterpreterOptions();
      // ..addDelegate(gpuDelegate);
      _interpreter = await Interpreter.fromAsset(
        'ml/model.tflite',
        options: interpreterOptions,
      );
    } catch (e) {
      print('Error loading model: $e');
    }
  }
}

class BoundingBoxPainter extends CustomPainter {
  final List<dynamic>? recognitions;
  final int imageHeight;
  final int imageWidth;

  BoundingBoxPainter(this.recognitions, this.imageHeight, this.imageWidth);

  @override
  void paint(Canvas canvas, Size size) {
    if (recognitions == null) return;
    final double factorX = size.width / imageWidth;
    final double factorY = size.height / imageHeight;

    for (var result in recognitions!) {
      final rect = result['rect'];
      final x = rect[0] * factorX;
      final y = rect[1] * factorY;
      final w = rect[2] * factorX;
      final h = rect[3] * factorY;
      final confidence = result['confidenceInClass'] * 100;

      final paint = Paint()
        ..color = Colors.red
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      canvas.drawRect(Rect.fromLTWH(x, y, w, h), paint);

      TextSpan span = TextSpan(
          style: TextStyle(color: Colors.white, fontSize: 12.0),
          text:
              '${result['detectedClass']} (${confidence.toStringAsFixed(1)}%)');
      TextPainter tp = TextPainter(
          text: span,
          textAlign: TextAlign.left,
          textDirection: ui.TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, Offset(x, y - 15));
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
