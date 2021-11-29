mkdir log
mkdir -p eval/eval_results
gdown https://drive.google.com/uc?id=1h3mZUMueGfrhPTZo1BoNNYLyPymlpZb7
unzip toydata_checkpoints.zip
rm toydata_checkpoints.zip
mv train_synthetic log/

gdown https://drive.google.com/uc?id=1EMdeJKc906Xn_RPWZSup-4s_TjNsp0IM
unzip shapenet_moreimgs_checkpoints.zip
rm shapenet_moreimgs_checkpoints.zip
mv more_imgs log/

gdown https://drive.google.com/uc?id=15ZUTzQAGtlJs-mUX0waY0pdnlLf3eO7n
unzip shapenet_moreshapes_checkpoints.zip
rm shapenet_moreshapes_checkpoints.zip
mv more_shapes log/