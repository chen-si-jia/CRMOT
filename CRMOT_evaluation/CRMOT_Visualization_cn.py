# 功能：对 CRMOT 的 CRTracker_In-domain 推理结果 进行可视化。保存图片和视频。
# 作者：陈思佳，华中科技大学
# 时间：20250304


import os
import shutil
import cv2
import colorsys


# 创建文件夹
def create_folder_if_not_exists(folder_path):
    """
    检查文件夹是否存在，如果不存在，则创建它。
    :param folder_path: 目标文件夹路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 递归创建文件夹
        print(f"文件夹 {folder_path} 创建成功")
    else:
        print(f"文件夹 {folder_path} 已存在")


# 清空文件夹中的内容
def clear_folder(folder_path):
    # 遍历文件夹中的所有内容
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # 如果是文件，删除文件
        if os.path.isfile(file_path):
            os.remove(file_path)
        # 如果是文件夹，递归删除
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


# ################################### 现在：框上ID，写的格式"ID：ID号码" #################################### #
def rectangle_bbox_score(image, x, y, w, h, color, thickness=2, label=None):
    """Draw a rectangle
    Parameters
    ----------
    label : Optional[str], id
    """
    
    pt1 = (int(x), int(y))
    pt2 = (int(x + w), int(y + h))
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        label = "ID:" + label # 改变label
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)
        
        pt1 = int(x), int(y - 5 - text_size[0][1]) # 修改一下起始点

        center = pt1[0] + 5, pt1[1] + text_size[0][1] - 2

        pt1 = int(x - 1), int(y - 3*text_size[0][1] - 10) # 修改一下起始点

        pt2 = pt1[0] + 3*text_size[0][0] - 10, pt1[1] + 3*text_size[0][1] + 10 # 修改一下结束点 

        cv2.rectangle(image, pt1, pt2, color, -4)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    2.5, (255, 255, 255), thickness) # 字体大小, 颜色, 粗细
        
        # print("label postion:", center)



if __name__ == '__main__':

    # 推理结果的txt文件
    sequence_dir_absolute_path = "/mnt/A/hust_csj/Code/Github/CRMOT/CRMOT_evaluation/data/In-domain/Inference_results/CRTracker_In-domain/Side_A man wearing a black coat and black trousers"
    
    for view in ["View1", "View2", "View3"]:

        sequence_absolute_path = os.path.join(sequence_dir_absolute_path, view + ".txt")
        
        # 读取推理结果的txt文件
        result_txt = []
        with open(sequence_absolute_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.strip().split(",")  # 去除换行符并按逗号分隔
                # 解析数据
                img_name = int(values[0])
                id = int(values[1])
                x1 = int(float(values[2]))
                y1 = int(float(values[3]))
                x2 = int(float(values[4]))
                y2 = int(float(values[5]))

                # xyxy -> xywh
                x = x1
                y = y1
                w = x2 - x1
                h = y2 - y1

                result_txt.append([img_name, id, x, y, w, h])

        # 获取该txt对应的语言描述
        description = sequence_dir_absolute_path.split("/")[-1] + "_" + view

        # 画完框的图片 的保存路径
        drawed_img_dir_path = os.path.join("/mnt/A/hust_csj/Code/Github/CRMOT/CRMOT_evaluation/Visualization/results", description, "img")

        # 创建文件夹
        create_folder_if_not_exists(drawed_img_dir_path) 
        # 若先前就存在，则清空该文件夹中的内容
        clear_folder(drawed_img_dir_path) 

        # 获取该txt对应的图片路径集合
        imgs_txt_path = os.path.join("/mnt/A/hust_csj/Code/Github/CRMOT/datasets/CRTrack/CRTrack_In-domain", "images", "test", sequence_dir_absolute_path.split("/")[-1].split("_")[0] + "_" + view, "img1")
        frame_names = os.listdir(imgs_txt_path)
        frame_names.sort()

        # 遍历所有图片
        for frame_name in frame_names:
            frame_name_path = os.path.join(imgs_txt_path, frame_name)

            # 读取图片
            image = cv2.imread(frame_name_path)

            if image is None:
                print("图片加载失败")
            
            # 在图片上绘制多个矩形框
            for result in result_txt:
                result_img_name = result[0]
                result_id = result[1]
                result_x = result[2]
                result_y = result[3]
                result_w = result[4]
                result_h = result[5]
                # 找到我们想要的图片对应的结果
                if (int(frame_name.split("_")[-1].split(".jpg")[0]) == result_img_name):
                    # 画矩形框
                    color = create_unique_color_uchar(result_id)
                    rectangle_bbox_score(image, result_x, result_y, result_w, result_h, color, thickness=2, label=str(result_id))

            # 保存修改后的图片
            drawed_img_path = os.path.join(drawed_img_dir_path, frame_name.split("/")[-1].split(".")[0] + ".jpg")
            cv2.imwrite(drawed_img_path, image)
        
        # =================  图像合并成视频  ====================
        # 图像文件夹路径
        folder_path = drawed_img_dir_path #换成自己图像的绝对路径

        # 画完框的视频 的保存路径
        drawed_videos_dir_path = os.path.join("/mnt/A/hust_csj/Code/Github/CRMOT/CRMOT_evaluation/Visualization/results", description, "video")
        # 创建文件夹
        create_folder_if_not_exists(drawed_videos_dir_path) 
        # 若先前就存在，则清空该文件夹中的内容
        clear_folder(drawed_videos_dir_path) 

        # 获取.jpg格式的图像文件列表
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        
        # 获取第一张图像的尺寸
        image_path = os.path.join(folder_path,image_files[0])
        first_image = cv2.imread(image_path)
        height,width,channels = first_image.shape
        
        # 设置视频输出路径和相关参数
        output_path = os.path.join(drawed_videos_dir_path, view + ".mp4")
        fps = 30.0 #帧率
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #编码器
        
        # 创建视频写入器对象
        video_writer = cv2.VideoWriter(output_path,fourcc,fps,(width,height))
        
        # 把图像逐帧写入视频
        for image_file in image_files:
            image_path = os.path.join(folder_path,image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)
        
        # 释放资源
        video_writer.release()
        
