package com.example.signlanguagetranslator;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.pm.PackageManager;

import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import com.example.signlanguagetranslator.databinding.ActivityMainBinding;
import com.example.signlanguagetranslator.ml.SignLanguageRecognitionModel;
import com.google.common.util.concurrent.ListenableFuture;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding viewBinding;
    private ExecutorService cameraExecutor;
    ProcessCameraProvider mCameraProvider;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        viewBinding = ActivityMainBinding.inflate(this.getLayoutInflater());
        setContentView(viewBinding.getRoot());

        Button btn = (Button)findViewById(R.id.btnCapture);

        mDefineCameraProvider();

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkAllPermissions()){
                    if (btn.getText().equals("Capture")) {
                        btn.setText("Stop");
                        startCamera();
                    }
                    else{
                        btn.setText("Capture");
                        stopCamera();
                    }
                }
                else{
                    requestPermissionLauncher.launch(Manifest.permission.CAMERA);
                }

            }
        });
        cameraExecutor = Executors.newSingleThreadExecutor();
    }

    private ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    Toast toast1 = Toast.makeText(this, "Camera access permission granted", Toast.LENGTH_SHORT);
                    toast1.show();
                } else {
                    Toast toast2 = Toast.makeText(this, "Camera access is required for the application to operate. Please enable camera.", Toast.LENGTH_SHORT);
                    toast2.show();
                }
            });

    private final boolean checkAllPermissions(){
        if (ContextCompat.checkSelfPermission(
                this.getBaseContext(), Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED)
            return true;
        return false;
    }

    private void mDefineCameraProvider(){
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                mCameraProvider = cameraProvider;
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void startCamera(){
        bindPreview(mCameraProvider);
    }

    private void stopCamera(){
        mCameraProvider.unbindAll();

    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        PreviewView previewView = findViewById(R.id.previewView);
        TextView translation = (TextView)findViewById(R.id.translation);


        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setTargetResolution(new Size(200, 200))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_BLOCK_PRODUCER)
                        .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        final String[] txt = {""};
        float[] arr = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        final int[] count = {0};
        final long[] start_time = {SystemClock.elapsedRealtime()};

        imageAnalysis.setAnalyzer(cameraExecutor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                Bitmap img = toBitmap(imageProxy);
                img = Bitmap.createScaledBitmap(img, 32, 32, false);

                ImageProcessor imageProcessor =
                        new ImageProcessor.Builder()
                                .add(new NormalizeOp(0, 255.0f))
                                .build();

                TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

                tensorImage.load(img);
                tensorImage = imageProcessor.process(tensorImage);

                ByteBuffer byteBuffer = tensorImage.getBuffer();
                Log.d("BB", "analyze: tensor image buf : " + Arrays.toString(byteBuffer.array()));

                try {
                    SignLanguageRecognitionModel model = SignLanguageRecognitionModel.newInstance(MainActivity.this);

                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    Log.d("INP", "input : " + Arrays.toString(inputFeature0.getFloatArray()));

                    SignLanguageRecognitionModel.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    int m = find_max(outputFeature0.getFloatArray());
                    Log.d("ML", "classification : " + m);
                    arr[m] += 1;
                    count[0] += 1;

                    // frame rate ~ nearly 30fps
                    if (count[0] == 60){
                        int m1 = find_max(arr);
                        String letter = getNearestClassification(m1);
                        if (!letter.equals("del")){
                            txt[0] = txt[0] + letter;
                        }
                        else txt[0] = txt[0].substring(0, txt[0].length() - 1);

                        translation.setText(txt[0]);
                        count[0] = 0;
                        start_time[0] = SystemClock.elapsedRealtime();

                        for (int i=0; i<29; i++){
                            arr[i] = 0;
                        }
                    }

                    model.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                // after done, release the ImageProxy object
                imageProxy.close();
            }

        });

        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview, imageAnalysis);
    }

    private Bitmap toBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer buffer = planes[0].getBuffer();
        int pixelStride = planes[0].getPixelStride();
        int rowStride = planes[0].getRowStride();
        int rowPadding = rowStride - pixelStride * image.getWidth();
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth()+rowPadding/pixelStride,
                image.getHeight(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(buffer);
        return bitmap;
    }


    private String getNearestClassification(int max){
        String[] categories = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "", " "};
        return categories[max];
    }

    private int find_max(float[] arr){
        Log.d("OUT", "find_max: output : " + Arrays.toString(arr));
        int max = 0;
        for (int i=0; i < arr.length; i++){
            if (arr[i] > arr[max]){
                max = i;
            }
        }
        return max;
    }

    private short[][] red;
    private short[][] green;
    private short[][] blue;

    /**
     * Map each intensity of an RGB colour into its respective colour channel
     */
    private void unpackPixel(int pixel, int row, int col) {
        red[row][col] = (short) ((pixel >> 16) & 0xFF);
        green[row][col] = (short) ((pixel >> 8) & 0xFF);
        blue[row][col] = (short) ((pixel >> 0) & 0xFF);

        Log.d("RGB", "unpackPixel: pixel values : " + red[row][col] + " " + green[row][col] + " " + blue[row][col]);
    }

    private Bitmap get_normalized_bitmap(Bitmap image){
        int newHeight = 32;
        int newWidth = 32;

        int[] pixels = new int[newWidth * newHeight];

        image.getPixels(pixels, 0, newWidth, 0, 0, newWidth, newHeight);

        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            int R = (color >> 16) & 0xff;
            int G = (color >> 8) & 0xff;
            int B = color & 0xff;
            int A = (color >> 24) & 0xff;

            float normalizedR = (float) R/255;
            float normalizedG = (float) G/255;
            float normalizedB = (float) B/255;
            float normalizedA = (float) A/255;

            Log.d("NORM", "normalized B : " + normalizedB);

            pixels[i] = ((int) (normalizedA * 255) << 24) | ((int) (normalizedR * 255) << 16) | ((int) (normalizedG * 255) << 8) | (int) (normalizedB * 255);
        }

        image.setPixels(pixels, 0, newWidth, 0, 0, newWidth, newHeight);
        Log.d("YYYY", "get_normalized_bitmap: pixels : " + Arrays.toString(pixels));


        return image;
    }

}