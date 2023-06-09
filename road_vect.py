# -*- coding: utf-8 -*-
"""
/***************************************************************************
 RoadVect
                                 A QGIS plugin
 This plugin vectorise roads
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-04-01
        git sha              : $Format:%H$
        copyright            : (C) 2023 by Isaeva Oksana
        email                : email
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import numpy as np
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QFileDialog
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from qgis import processing
from .additional.ProcessRoads import ProcessRoads
import time

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .road_vect_dialog import RoadVectDialog
import os.path


class RoadVect:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'RoadVect_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&Road Vectorisation')

        projectPath = os.path.dirname(os.path.abspath(__file__))

        self.processRoads = ProcessRoads(projectPath)

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

        self.layers = None
        self.index_selected_layer = -1

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('RoadVect', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/road_vect/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Vectorise roads'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&Road Vectorisation'),
                action)
            self.iface.removeToolBarIcon(action)

    def errorWindow(self, info):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(info)
        msg.setWindowTitle("Error")
        msg.exec_()

    def selectOutputFile(self):
        filename = QFileDialog.getSaveFileName(self.dlg, "Выберете путь для сохранения файла ", "", '*.shp')
        if filename:
            self.dlg.lineEditSave.setText(filename[0])

    def comboBoxLChanged(self):
        """Selected layer changed event"""
        self.index_selected_layer = self.dlg.comboBoxLayer.currentIndex() - 1

    def run(self):
        """Run method that performs all the real work"""

        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = RoadVectDialog()
            self.dlg.pushButtonClose.clicked.connect(self.closeDialog)
            self.dlg.pushButtonStart.clicked.connect(self.startProcess)
            self.dlg.comboBoxLayer.activated.connect(self.comboBoxLChanged)
            self.dlg.toolButton.clicked.connect(self.selectOutputFile)

        # Adding menu items
        self.layers = [layer for layer in QgsProject.instance().mapLayers().values()]
        names = [layer.name() for layer in self.layers]
        names.insert(0, "")
        self.dlg.comboBoxLayer.clear()
        self.dlg.comboBoxLayer.addItems(names)

        # init indexes
        if self.index_selected_layer >= 0:
            self.dlg.comboBoxLayer.setCurrentIndex(self.index_selected_layer + 1)

        # show the dialog
        self.dlg.show()
        # Run the dialog event loop
        result = self.dlg.exec_()
        # See if OK was pressed
        if result:
            print("done")
            pass

    def closeDialog(self):
        self.dlg.done(0)

    def startProcess(self):
        print("start")

        if self.index_selected_layer < 0:
            self.errorWindow("Слой не выбран")
            return

        if not type(self.layers[self.index_selected_layer]) == QgsRasterLayer:
            self.errorWindow("Выбранный слой не является растром")
            return

        if self.dlg.lineEditSave.text():
            shp_save_path = self.dlg.lineEditSave.text()
        else:
            self.errorWindow("Введите путь для сохранения")
            return

        try:
            save_all = self.dlg.checkBoxSaveTemp.isChecked()
            open_at_the_end = self.dlg.checkBoxOpen.isChecked()
            selected_layer = self.layers[self.index_selected_layer]

            selected_layer_path = selected_layer.dataProvider().dataSourceUri()

            # binarize image
            imgpath_segmented = self.processRoads.processModule(selected_layer_path, self.dlg.progressBar)

            # vectorise image
            img_name = imgpath_segmented.split('/')[-1]
            self.processRoads.vectorise(img_name, imgpath_segmented)

            # save shp with offset
            offset = [selected_layer.dataProvider().extent().xMinimum(),
                      selected_layer.dataProvider().extent().yMaximum()]
            shp_file = self.processRoads.save_vector(img_name, shp_save_path, offset)

            # delete temp objects
            '''
                There is a problem with deleting additional temp shp files
                so for now deleting will ignore errors
            '''
            temp_dir = self.processRoads.get_working_dir(img_name)
            if not save_all:
                time.sleep(1)
                self.processRoads.clean_up(img_name)
            else:
                os.startfile(temp_dir)

            # open shp layer
            if open_at_the_end:
                vlayer = QgsVectorLayer(shp_save_path, shp_file, "ogr")
                if not vlayer.isValid():
                    print("Layer failed to load!")
                else:
                    QgsProject.instance().addMapLayer(vlayer)

        except Exception as e:
            self.errorWindow(e)
