# 安裝 opencv_cuda
https://medium.com/@tcfsh210119/ubuntu-opencv-gpu%E5%AE%89%E8%A3%9D-df8a9563343c

# 執行程式
```
sudo apt-get install redis-server
conda activate your_env
pip install -r requirements.txt
python worker.py & python app.py
```

# 下載各個資料夾並放到 electronic_pet2.0/ 中
data/：放 P-learning 之 dataset
https://drive.google.com/file/d/1Z1IxsaHORZHOf6OEfCjwCvvWnI9B6LiH/view?usp=sharing

model/：放 P-learning 產生之 encoder
https://drive.google.com/file/d/1pfMH7rJZL1dAt7e2RO8DuhtUXnFv2grK/view?usp=sharing

video_test/：放欲測試之video, 在此產生output檔
https://drive.google.com/file/d/1u9RHe6FnVcWjiLNImp0eTD7Vgwi5WnDN/view?usp=sharing

yolo/：放訓練好的 yolo model
https://drive.google.com/file/d/1yChpfpXLKL52T7XO5O1pIzF7yes0egLT/view?usp=sharing

tmp_unseen/：存放 unseen class image (會自動產生)
