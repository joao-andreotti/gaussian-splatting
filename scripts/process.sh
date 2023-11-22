mkdir _tmp;
mkdir $2;
mkdir _tmp/images;
ffmpeg -i $1 _tmp/images/%04d.jpg;
colmap automatic_reconstructor --workspace_path ./_tmp --image_path ./_tmp/images;
colmap model_converter --input_path ./_tmp/sparse/0 --output_path ./$2/ --output_type TXT;
rm -r _tmp;
