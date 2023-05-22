import os
import pathlib
import configparser
import shutil
import sys
import tempfile
import subprocess
import time
from PyQt5.QtWidgets import QProgressBar
from .ImageWorker import ImageWorker
from .RoadsVectorization import RoadsVectorization
#from . import model

from contextlib import contextmanager

@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

class ProcessRoads:
    __dir_model = ""
    __full_path = ""
    __config = None
    __model_name = ""
    __dir_temp_results = ""
    __segmented_tag = '-segmented'

    @property
    def model_name(self):
        return self.__model_name

    def __init__(self, full_path):
        self.__full_path = full_path + '/'
        self.__dir_model = self.__full_path + 'model/'
        self.__dir_temp_results = self.__full_path + "temp/"
        self.readConfig()

    def readConfig(self):
        directory = self.__dir_model
        file = "settings.ini"
        config = configparser.ConfigParser()
        config.read(pathlib.Path(directory).rglob(file), encoding="utf-8")
        self.__config = config
        self.__model_name = config["COMMON"]["NAME"]

    def getInfo(self, type="COMMON"):
        return self.__config[type]

    def processModule(self, image_path, progressbar: QProgressBar):
        progressbar.setValue(0)
        info = self.getInfo("SYSTEM")
        image_w = ImageWorker(image_path)
        python_bin = sys.executable.rsplit('\\', 1)[0]+"\\python3.exe"
        script_file = self.__full_path + "additional/model.py"

        directory = os.path.join(self.__dir_temp_results, image_path.split('/')[-1].split('.')[-0] + self.__segmented_tag)
        img_ext = image_path.split('.')[-1]
        image_w.preprocessImage(int(info["HEIGHT"]), int(info["WIDTH"]), img_ext,
                               os.path.join(directory, info["FOLDER_TILES"]))

        dir_tiles_segmented = os.path.join(directory, info["FOLDER_TILES_SEGMENTED"])
        if not os.path.exists(dir_tiles_segmented):
            os.makedirs(dir_tiles_segmented)

        p = subprocess.Popen([python_bin, script_file],
                             cwd=directory,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             shell=True)

        while p.poll() == None:
            percent = image_w.checkProgress(os.path.join(directory, info["FOLDER_TILES_SEGMENTED"]))
            progressbar.setValue(percent)
            if percent == 100:
                break
            time.sleep(1)

        imgpath_segmented = image_w.processTiles(int(info["HEIGHT"]), int(info["WIDTH"]),
                                                os.path.join(directory, info["FOLDER_TILES_SEGMENTED"]),
                                                self.__dir_temp_results, self.__segmented_tag)
        progressbar.setValue(100)
        return imgpath_segmented

    def vectorise(self, img_name_ext, dir_image):
        return RoadsVectorization().vectorise(img_name_ext, dir_image, self.__dir_temp_results)

    def save_vector(self, img_name_ext, path_to_save, offset=None):
        path = os.path.join(self.__dir_temp_results, os.path.splitext(img_name_ext)[0])
        if os.path.exists(path):
            return RoadsVectorization().save_shp(path, path_to_save, offset)

    def get_working_dir(self, img_name_ext):
        path = os.path.join(self.__dir_temp_results, os.path.splitext(img_name_ext)[0])
        if (os.path.exists(path)):
            return path
        else:
            return None

    def clean_up(self, img_name_ext):
        '''Удаление временных файлов'''
        path = os.path.join(self.__dir_temp_results, os.path.splitext(img_name_ext)[0])
        path_file = os.path.join(self.__dir_temp_results, img_name_ext)
        if (os.path.exists(path)):
            shutil.rmtree(path, ignore_errors=True)
            os.remove(path_file)