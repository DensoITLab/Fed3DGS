# Copyright (C) 2024 Denso IT Laboratory, Inc.
# All Rights Reserved
echo "create database"
python tools/create_db.py  -out $1 -r $2 --index-file $3
colmap feature_extractor --database_path $1/database.db --image_path $2/rgbs --image_list_path $3 --ImageReader.camera_model PINHOLE
colmap exhaustive_matcher --database_path $1/database.db
mkdir $1/sparse/0
colmap point_triangulator --database_path $1/database.db --image_path $2/rgbs --input_path $1/sparse --output_path $1/sparse/0 --Mapper.tri_ignore_two_view_tracks=0
rm -r $1/sparse/*.txt
rm $1/database.db
