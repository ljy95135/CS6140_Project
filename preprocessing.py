"""
I will use annotations of Standford dog images to crop images
an alternative method is extracting faces

When only use one kind of dog, the image number may not be enough, so data augmentation (flip and rotate) will be used.
"""
import glob
import os
from PIL import Image
from xml.dom import minidom
from scipy import misc


def face_extracting(files):
    """
    Note: Only this method need to run under dlib_face env

    Use dlib and opencv to extract dog faces
    use another python environment for it is hard to install dlib on windows
    not necessary since result is in data

    need to mannually filter human face
    """
    import dlib  # make sure it is installed
    import cv2
    for file in files:
        detector = dlib.get_frontal_face_detector()
        img = cv2.imread(file)
        b, g, r = cv2.split(img)
        img_rgb = cv2.merge([r, g, b])

        dets = detector(img_rgb, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for index, face in enumerate(dets):
            width = face.right() - face.left()
            height = face.bottom() - face.top()
            left = max(0, face.left() - int(width * 0.5))
            right = min(face.right() + int(width * 0.5), img.shape[1])
            top = max(0, face.top() - int(height * 0.7))
            bottom = min(face.bottom() + int(height * 0.5), img.shape[0])
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index, left, top, right, bottom))

            cv2.rectangle(img, (left, top), (right, bottom),
                          (0, 255, 0), 6)
            cropImg = img[int(top + 4):int(bottom - 4),
                      int(left + 4):int(right - 4)]
            file_name = file.split('\\')[-1]
            cv2.imwrite('./tmp/result_' + file_name, cropImg)


def crop_via_annotation(directory):
    """
    It will crop images based on annotation inside stanford dogs dataset
    To run preprocessing, dir should follow this format:
    dir/Images/
    dir/Annotation/
    """
    img_dir = os.path.join(directory, 'Images')
    annotation_dir = os.path.join(directory, 'Annotation')
    image_names = os.listdir(img_dir)
    for image_name in image_names:
        annotation = os.path.join(annotation_dir, image_name.split('.')[0])
        annotation_xml = minidom.parse(annotation)
        xmin = int(annotation_xml.getElementsByTagName('xmin')[0].firstChild.nodeValue)
        ymin = int(annotation_xml.getElementsByTagName('ymin')[0].firstChild.nodeValue)
        xmax = int(annotation_xml.getElementsByTagName('xmax')[0].firstChild.nodeValue)
        ymax = int(annotation_xml.getElementsByTagName('ymax')[0].firstChild.nodeValue)
        image = misc.imread(os.path.join(img_dir, image_name))
        crop_image = image[ymin:ymax, xmin: xmax, :]
        misc.imsave(os.path.join(directory, 'cropped_' + image_name), crop_image)


def flip(files):
    """
    files are in tmp directory
    output files are flip_<name>.jpg in tmp
    remove them back to data/
    """
    for file in files:
        file_name = file.split('\\')[-1]
        img = Image.open(file, 'r')
        img.load()
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_img.save('./tmp/flip_' + file_name, 'JPEG', quality=100)


def rotate(files):
    """
    files are in tmp directory
    output files are rotate_degree_<name>.jpg in tmp
    remove them back to data/
    """
    for file in files:
        file_name = file.split('\\')[-1]
        img = Image.open(file, 'r')
        img.load()
        for degree in [-11, -7, -3, 3, 7, 11]:
            rotate_img = img.rotate(degree)
            rotate_img.save('./tmp/rotate_' + get_degree_str(degree) + file_name, 'JPEG', quality=100)


def get_degree_str(degree):
    return str(degree) if degree > 0 else 'm' + str(-degree)


if __name__ == '__main__':
    # uncomment the operation you want to run and make sure file is put into right path

    # face extracting
    # files = glob.glob(os.path.join('./tmp/', '*.*'))
    # face_extracting(files)

    # augment images (flip and rotate)
    # files = glob.glob(os.path.join('./tmp/', '*.*'))
    # flip(files)
    # rotate(files)

    # crop via annotation
    # dirs = ['./tmp/n02085620-Chihuahua', './tmp/n02085936-Maltese_dog', './tmp/n02096585-Boston_bull']
    # for dir in dirs:
    #     crop_via_annotation(dir)

    pass
