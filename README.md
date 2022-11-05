


1.  pixel_nerf data folder structure

```
data_folder
    |
    |
    |____ images_folder
    |         |
    |         |____ 01.npy
    |         |____ 02.npy
    |         |____ 03.npy
    .              :      
    .              :      
    .              :      
    .              :      
    |
    |
    |___ objs.txt
    |
    |___poses.npz


```
### run pixelnerf

```
python3 run_pixelnerf.py <data_folder_path> <source_image> <target_image> <obj_id>
```



### run nerf

```
python3 run_nerf.py <npz_data_file_path>
```


