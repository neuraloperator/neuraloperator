# Adapted from https://stackoverflow.com/questions/59111396/loop-through-the-directories-to-get-the-image-files
# Usage 
# chmod 755 compress_images.bash
# bash compress_images.sh ./_static/images

find "$1" \( -iname \*.png \) -print0 | while read -r -d $'\0' file; do
  # base="${file##*/}"  $base is the file name with all the directory stuff stripped off
  # dir="${file%/*}"    $dir is the directory with the file name stripped off
  # target="${file%.*}" This is the full name without extension
  target="${file%.*}.jpg"
  sips -s format jpeg -s formatOptions low ${file} --out ${target}
  git rm ${file}
done
